package sq.flutter.tflite;

import static android.os.Environment.DIRECTORY_DCIM;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.Color;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Environment;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;

import androidx.annotation.RequiresApi;

import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBufferFloat;


public class TflitePlugin implements MethodCallHandler {
  private final Registrar mRegistrar;
  private Interpreter tfLite;
  private boolean tfLiteBusy = false;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][][][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;

  private ImageProcessor imageProcessor;


  public static void registerWith(Registrar registrar) {
    final MethodChannel channel = new MethodChannel(registrar.messenger(), "tflite");
    channel.setMethodCallHandler(new TflitePlugin(registrar));
  }

  private TflitePlugin(Registrar registrar) {
    this.mRegistrar = registrar;
  }

  @Override
  public void onMethodCall(MethodCall call, Result result) {
    if (call.method.equals("loadModel")) {
      try {
        String res = loadModel((HashMap) call.arguments);
        result.success(res);
      } catch (Exception e) {
        result.error("Failed to load model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnImage")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnBinary")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runModelOnFrame")) {
      try {
        new RunModelOnFrame((HashMap) call.arguments, result).executeTfliteTask();
      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnImage")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnBinary")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("detectObjectOnFrame")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("close")) {
      close();
    } else if (call.method.equals("runPix2PixOnImage")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runPix2PixOnBinary")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runPix2PixOnFrame")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runSegmentationOnImage")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runSegmentationOnBinary")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runSegmentationOnFrame")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runPoseNetOnImage")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runPoseNetOnBinary")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else if (call.method.equals("runPoseNetOnFrame")) {
      try {

      } catch (Exception e) {
        result.error("Failed to run model", e.getMessage(), e);
      }
    } else {
      result.error("Invalid method", call.method.toString(), "");
    }
  }

  private String loadModel(HashMap args) throws IOException {
    String model = args.get("model").toString();
    Object isAssetObj = args.get("isAsset");
    boolean isAsset = isAssetObj == null ? false : (boolean) isAssetObj;
    MappedByteBuffer buffer = null;
    String key = null;
    AssetManager assetManager = null;
    if (isAsset) {
      assetManager = mRegistrar.context().getAssets();
      key = mRegistrar.lookupKeyForAsset(model);
      AssetFileDescriptor fileDescriptor = assetManager.openFd(key);
      FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    } else {
      FileInputStream inputStream = new FileInputStream(new File(model));
      FileChannel fileChannel = inputStream.getChannel();
      long declaredLength = fileChannel.size();
      buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, declaredLength);
    }

    int numThreads = (int) args.get("numThreads");
    Boolean useGpuDelegate = (Boolean) args.get("useGpuDelegate");
    if (useGpuDelegate == null) {
      useGpuDelegate = false;
    }

    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    if (useGpuDelegate){
      GpuDelegate delegate = new GpuDelegate();
      tfliteOptions.addDelegate(delegate);
    }
    tfLite = new Interpreter(buffer, tfliteOptions);

    String labels = args.get("labels").toString();

    key = mRegistrar.lookupKeyForAsset(labels);
    loadLabels(assetManager, key);
//    if (labels.length() > 0) {
//      if (isAsset) {
//        key = mRegistrar.lookupKeyForAsset(labels);
//        loadLabels(assetManager, key);
//      } else {
//        loadLabels(null, labels);
//      }
//    }

    return "success";
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      if (assetManager != null) {
        br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      } else {
        br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(path))));
      }
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()][60][17];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file", e);
    }
  }

  private float[] GetTopN(int numResults, float threshold) {
    // PriorityQueue<Map<String, Object>> pq =
    //     new PriorityQueue<>(
    //         1,
    //         new Comparator<Map<String, Object>>() {
    //           @Override
    //           public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
    //             return Float.compare((float) rhs.get("confidence"), (float) lhs.get("confidence"));
    //           }
    //         });



    // for (int i = 0; i < labels.size(); ++i) {
    //   // float confidence = labelProb[0][i];
    //   float confidence = 1f;
    //   if (confidence > threshold) {
    //     Map<String, Object> res = new HashMap<>();
    //     res.put("index", i);
    //     res.put("label", labelProb[0][i][0][0]);
    //     res.put("confidence", confidence);
    //     res.put("raw",labelProb);
    //     pq.add(res);
    //   }
    // }




    //final ArrayList<Map<String, Object>> recognitions = new ArrayList<>();
    // int recognitionsSize = Math.min(pq.size(), numResults);
    // for (int i = 0; i < recognitionsSize; ++i) {
    //   recognitions.add(pq.poll());
    // }

    // Map<String, float[][][][]> res = new HashMap<>();

    final float[] recognitions = new float[48];
    // res.put("result",labelProb);


    for(int i =0; i< 17; i++ ) {
      float max = 0;
      int locationX =0;
      int locationY =0;

      for(int j =0; j<60; j++) {
        for(int k=0; k<80; k ++) {
          if(  labelProb[0][k][j][i]>max) {
            //System.out.println("labelprob: "+labelProb[0][k][j][i]);
            max = labelProb[0][k][j][i];
            locationX = j;
            locationY = k;
          }
        }
      }
      recognitions[2*i] = locationX;
      recognitions[2*i+1] = locationY;
    }

    return recognitions;
  }
  Bitmap feedOutput(ByteBuffer imgData, float mean, float std) {
    Tensor tensor = tfLite.getOutputTensor(0);
    int outputSize = tensor.shape()[1];
    Bitmap bitmapRaw = Bitmap.createBitmap(outputSize, outputSize, Bitmap.Config.ARGB_8888);

    System.out.print(tensor.dataType());

    if (tensor.dataType() == DataType.FLOAT32) {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 16);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF) << 8);
          pixelValue |= ((Math.round(imgData.getFloat() * std + mean) & 0xFF));
          double k = Math.random();
          String s = String.valueOf(k);
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    } else {
      for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
          int pixelValue = 0xFF << 24;
          pixelValue |= ((imgData.get() & 0xFF) << 16);
          pixelValue |= ((imgData.get() & 0xFF) << 8);
          pixelValue |= ((imgData.get() & 0xFF));
          bitmapRaw.setPixel(j, i, pixelValue);
        }
      }
    }



    return bitmapRaw;
  }

  boolean createTodoOneImage = false;
  boolean createTodoOneImage1 = false;

  @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
  ByteBuffer feedInputTensorFrame(byte[] bytesList, int imageHeight, int imageWidth, float mean, float std, int rotation) throws IOException {
//    ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
//    ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
//    ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));
//
//
//
//    int Yb = Y.remaining();
//    int Ub = U.remaining();
//    int Vb = V.remaining();
//
//    byte[] data = new byte[Yb + Ub + Vb];
//
//    Y.get(data, 0, Yb);
//    V.get(data, Yb, Vb);
//    U.get(data, Yb + Vb, Ub);

    Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
//    Allocation bmData = renderScriptNV21ToRGBA888(
//            mRegistrar.context(),
//            imageWidth,
//            imageHeight,
//            data);
//    bmData.copyTo(bitmapRaw);
    new YuvToRgbConverter(mRegistrar.context()).yuvToRgb(bytesList,bitmapRaw);


    if(createTodoOneImage1 == false) {
      try {
        bitmapToFile(bitmapRaw,"mora_before_normal");
      } catch (IOException e) {
        e.printStackTrace();
      }
      createTodoOneImage1 = true;
    }

    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);

    if(imageProcessor == null){
      imageProcessor = buildImageProcessor(320,240,mean,std);
    }

    TensorImage inputImageBuffer = new TensorImage(tfLite.getInputTensor(0).dataType());


    inputImageBuffer.load(bitmapRaw);

    float[] values = inputImageBuffer.getTensorBuffer().getFloatArray();
   
    for(int i = 0; i < values.length; ++i) {
      values[i] = (values[i] - 116.78f);
    }


    inputImageBuffer.load(values, inputImageBuffer.getTensorBuffer().getShape());

//    for(int x =0; x<240; x++) {
//      for(int y=0; y<320; y++){
//        int p = bitmapRaw.getPixel(x,y);
//        int A = (p >>24) & 0xff;
//        int R = (p >> 16) & 0xff;
//        int G = (p >> 8) & 0xff;
//        int B = p & 0xff;
//        int R_ = R -124;
//        int G_ = G -117;
//        int B_ = B -104;
//        p = (A << 24) + (R_ << 16) + (G_ <<8) + B_;
//        bitmapRaw.setPixel(x,y,p);
//      }
//    }
//    inputImageBuffer.load(bitmapRaw);
    imageProcessor.process(inputImageBuffer);

    //TODO save image data
    if(createTodoOneImage == false) {
      try {
        bitmapToFile(inputImageBuffer.getBitmap(),"mora_after_normal");
      } catch (IOException e) {
        e.printStackTrace();
      }
      createTodoOneImage = true;
    }

    return inputImageBuffer.getBuffer();

//    return feedInputTensor(bitmapRaw, mean, std);
  }

  public Allocation renderScriptNV21ToRGBA888(Context context, int width, int height, byte[] nv21) {
    // https://stackoverflow.com/a/36409748
    RenderScript rs = RenderScript.create(context);
    ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

    Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
    Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

    Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
    Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

    in.copyFrom(nv21);

    yuvToRgbIntrinsic.setInput(in);
    yuvToRgbIntrinsic.forEach(out);
    return out;
  }

  private abstract class TfliteTask extends AsyncTask<Void, Void, Void> {
    Result result;
    boolean asynch;

    TfliteTask(HashMap args, Result result) {
      if (tfLiteBusy) throw new RuntimeException("Interpreter busy");
      else tfLiteBusy = true;
      Object asynch = args.get("asynch");
      this.asynch = asynch == null ? false : (boolean) asynch;
      this.result = result;
    }

    abstract void runTflite();

    abstract void onRunTfliteDone();

    public void executeTfliteTask() {
      if (asynch) execute();
      else {
        runTflite();
        tfLiteBusy = false;
        onRunTfliteDone();
      }
    }

    protected Void doInBackground(Void... backgroundArguments) {
      runTflite();
      return null;
    }

    protected void onPostExecute(Void backgroundResult) {
      tfLiteBusy = false;
      onRunTfliteDone();
    }
  }

  private class RunModelOnFrame extends TfliteTask {
    int NUM_RESULTS;
    float THRESHOLD;
    long startTime;
    ByteBuffer imgData;

    RunModelOnFrame(HashMap args, Result result) throws IOException {
      super(args, result);
      List<byte[]> bytesList = (ArrayList) args.get("bytesList");
      double mean = (double) (args.get("imageMean"));
      float IMAGE_MEAN = (float) mean;
      double std = (double) (args.get("imageStd"));
      float IMAGE_STD = (float) std;
      int imageHeight = (int) (args.get("imageHeight"));
      int imageWidth = (int) (args.get("imageWidth"));
      int rotation = (int) (args.get("rotation"));
      NUM_RESULTS = (int) args.get("numResults");
      double threshold = (double) args.get("threshold");
      THRESHOLD = (float) threshold;

      startTime = SystemClock.uptimeMillis();

      imgData = feedInputTensorFrame(bytesList, imageHeight, imageWidth, IMAGE_MEAN, IMAGE_STD, rotation);
    }

    protected void runTflite() {
      tfLite.run(imgData, labelProb);
    }

    protected void onRunTfliteDone() {
      Log.v("time", "Inference took " + (SystemClock.uptimeMillis() - startTime));
      result.success(GetTopN(NUM_RESULTS, THRESHOLD));
    }
  }

  private void close() {
    if (tfLite != null)
      tfLite.close();
    labels = null;
    labelProb = null;
  }

  private File bitmapToFile( Bitmap bitmap, String fileName) throws IOException {
    OutputStream out= null;

    File file = new File(Environment.getExternalStoragePublicDirectory(DIRECTORY_DCIM)+"/mora/"+fileName +".jpg");

    try{
      file.getParentFile().mkdirs();
      Log.i("bitmapToFile","Path : "+ file.getPath().toString());
      if(file.isFile()){
        file.delete();
      }
      Log.i("error","1");
      file.createNewFile();
      Log.i("error","2");
      out = new FileOutputStream(file);
      Log.i("error","3");
      bitmap.compress(Bitmap.CompressFormat.JPEG, 80, out);
    }finally{
      //out.close();
    }
    return file;
  }

  private ImageProcessor buildImageProcessor(int inputImageHeight,int inputImageWidth,float mean, float stddev) {
    int resizeHeight = Math.max(inputImageWidth, inputImageHeight);
    int resizeWidth = Math.min(inputImageWidth, inputImageHeight);
    final float[] means;
    final float[] stddevs;
    means = new float[3];
    stddevs = new float[3];
//    means[0] = 123.68f;
//    means[1] = 116.78f;
//    means[2] = 103.94f;
    means[0] = 123.68f;
    means[1] = 116.78f;
    means[2] = 103.94f;

    stddevs[0] = 1f;
    stddevs[1] = 1f;
    stddevs[2] = 1f;

    System.out.println("nomalize op is run");
    return new ImageProcessor.Builder()
            .add(new Rot90Op(2))
            .add(new NormalizeOp(means,stddevs))
            .build();

    //.add(new ResizeOp(320, 240, ResizeOp.ResizeMethod.BILINEAR))
  }
}
