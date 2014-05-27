/*
 * Copyright 2013 Sony Corporation
 */

package com.example.sony.cameraremote;


import android.annotation.TargetApi;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.media.FaceDetector;
import android.media.FaceDetector.Face;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import com.example.sony.cameraremote.utils.SimpleLiveviewSlicer;
import com.example.sony.cameraremote.utils.SimpleLiveviewSlicer.Payload;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * A SurfaceView based class to draw liveview frames serially.
 */
public class SimpleLiveviewSurfaceView extends SurfaceView implements
		SurfaceHolder.Callback {

	private static final String TAG = SimpleLiveviewSurfaceView.class
			.getSimpleName();

	private SimpleRemoteApi mRemoteApi;
	private boolean mWhileFetching;
	private final BlockingQueue<byte[]> mJpegQueue = new ArrayBlockingQueue<byte[]>(
			2);
	private final boolean mInMutableAvailable = Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB;
	private Thread mDrawerThread;
	private int mPreviousWidth = 0;
	private int mPreviousHeight = 0;
	private final Paint mFramePaint;
	public byte[] Img;
	public boolean running = false;
	public static Thread tracker;

	/**
	 * Contractor
	 * 
	 * @param context
	 */
	public SimpleLiveviewSurfaceView(Context context) {
		super(context);
		getHolder().addCallback(this);
		mFramePaint = new Paint();
		mFramePaint.setDither(true);
	}

	/**
	 * Contractor
	 * 
	 * @param context
	 * @param attrs
	 */
	public SimpleLiveviewSurfaceView(Context context, AttributeSet attrs) {
		super(context, attrs);
		getHolder().addCallback(this);
		mFramePaint = new Paint();
		mFramePaint.setDither(true);
	}

	/**
	 * Contractor
	 * 
	 * @param context
	 * @param attrs
	 * @param defStyle
	 */
	public SimpleLiveviewSurfaceView(Context context, AttributeSet attrs,
			int defStyle) {
		super(context, attrs, defStyle);
		getHolder().addCallback(this);
		mFramePaint = new Paint();
		mFramePaint.setDither(true);
	}

	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width,
			int height) {
		// do nothing.
	}

	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		// do nothing.
	}
	
	

	@Override
	public void surfaceDestroyed(SurfaceHolder holder) {
		mWhileFetching = false;
	}

	/**
	 * Bind a Remote API object to communicate with Camera device. Need to call
	 * this method before calling start() method.
	 * 
	 * @param remoteApi
	 */
	public void bindRemoteApi(SimpleRemoteApi remoteApi) {
		mRemoteApi = remoteApi;
	}

	/**
	 * Start retrieving and drawing liveview frame data by new threads.
	 * 
	 * @return true if the starting is completed successfully, false otherwise.
	 * @exception IllegalStateException
	 *                when Remote API object is not set.
	 * @see SimpleLiveviewSurfaceView#bindRemoteApi(SimpleRemoteApi)
	 */
	public boolean start() {
		if (mRemoteApi == null) {
			throw new IllegalStateException("RemoteApi is not set.");
		}
		if (mWhileFetching) {
			Log.w(TAG, "start() already starting.");
			return false;
		}

		mWhileFetching = true;

		// A thread for retrieving liveview data from server.
		new Thread() {
			@Override
			public void run() {
				Log.d(TAG, "Starting retrieving liveview data from server.");
				SimpleLiveviewSlicer slicer = null;

				try {
					// Prepare for connecting.
					JSONObject replyJson = null;

					replyJson = mRemoteApi.startLiveview();
					if (!isErrorReply(replyJson)) {
						JSONArray resultsObj = replyJson.getJSONArray("result");
						String liveviewUrl = null;
						if (1 <= resultsObj.length()) {
							// Obtain liveview URL from the result.
							liveviewUrl = resultsObj.getString(0);
						}
						if (liveviewUrl != null) {
							// Create Slicer to open the stream and parse it.
							slicer = new SimpleLiveviewSlicer();
							slicer.open(liveviewUrl);
						}
					}

					if (slicer == null) {
						mWhileFetching = false;
						return;
					}

					while (mWhileFetching) {
						final Payload payload = slicer.nextPayload();
						if (payload == null) { // never occurs
							Log.e(TAG, "Liveview Payload is null.");
							continue;
						}

						if (mJpegQueue.size() == 2) {
							mJpegQueue.remove();
						}
						Img = payload.jpegData;
						mJpegQueue.add(payload.jpegData);

					}
				} catch (IOException e) {
					Log.w(TAG, "IOException while fetching: " + e.getMessage());
				} catch (JSONException e) {
					Log.w(TAG, "JSONException while fetching");
				} finally {
					// Finalize
					try {
						if (slicer != null) {
							slicer.close();
						}
						mRemoteApi.stopLiveview();
					} catch (IOException e) {
						Log.w(TAG,
								"IOException while closing slicer: "
										+ e.getMessage());
					}

					if (mDrawerThread != null) {
						mDrawerThread.interrupt();
					}

					mJpegQueue.clear();
					mWhileFetching = false;
				}
			}
		}.start();

		// A thread for drawing liveview frame fetched by above thread.
		/*
		 * mDrawerThread = new Thread() {
		 * 
		 * @Override public void run() { Log.d(TAG,
		 * "Starting drawing liveview frame."); Bitmap frameBitmap = null;
		 * 
		 * BitmapFactory.Options factoryOptions = new BitmapFactory.Options();
		 * factoryOptions.inSampleSize = 1; if (mInMutableAvailable) {
		 * initInBitmap(factoryOptions); }
		 * 
		 * while (mWhileFetching) { try { byte[] jpegData = mJpegQueue.take();
		 * frameBitmap = BitmapFactory.decodeByteArray(jpegData, 0,
		 * jpegData.length, factoryOptions); } catch (IllegalArgumentException
		 * e) { if (mInMutableAvailable) { clearInBitmap(factoryOptions); }
		 * continue; } catch (InterruptedException e) { Log.i(TAG,
		 * "Drawer thread is Interrupted."); break; }
		 * 
		 * if (mInMutableAvailable) { setInBitmap(factoryOptions, frameBitmap);
		 * } //drawFrame(frameBitmap); }
		 * 
		 * if (frameBitmap != null) { frameBitmap.recycle(); } mWhileFetching =
		 * false; } }; mDrawerThread.start();
		 */
		new Thread() {
			private Mat filt_freq;
			private int R, C, lx, ly, frames;
			private double PSR;
			private boolean found;
			private Mat gi;
			private int G_size = 6;
			public  Mat[] Gallery = new Mat[G_size];
			public  int Detect = -1;
			public  int person = -1;
			public  String[] Names = new String[G_size];
			public  boolean recon = false;
			private Size filtersize = new Size(64,64);
			//private double[] buf = new double[64*64*2];
			//private double[] buf1 = new double[64*64*2];
			private double[] normbuf = new double[64*64];
			private double[] normbuf2 = new double[64*64];
			private double[] buf4 = new double[64*64];
			//private double[] buf5 = new double[64*64];
			//private double[] buf6 = new double[64*64];
					
			@Override
			public void run() {
				if (!running) {
					 gi = new Mat(64,64, CvType.CV_64F, Scalar.all(0));

					for(int i = 0; i < 64; i++){
						for(int j=0; j < 64; j++){
							double q = (-1 * (Math.pow((i+1-Math.round(64/2.0)), 2) + Math.pow((j+1-Math.round(64/2.0)), 2)))/350;
							q = Math.pow(Math.E, q);
							gi.put(i, j, q);
						}
					}
					final String orgName = Thread.currentThread().getName();
					Thread.currentThread().setName(orgName + " - TRACKING");
					found = false;
					Mat Training[] = new Mat[50];
					int missedframes = 0;
					int frameupdate = 0;
					R = 0;
					C = 0;
					lx = 0;
					ly = 0;
					frames = 0;
					int framecollect = 20;
					PSR = 0;
					Mat FiltLearn = new Mat();
					Mat[] ToP = new Mat[1];
					Mat avg = new Mat();
					Mat img = null;
					Mat real = null;
					ArrayList<Mat> channelz = new ArrayList<Mat>();
					Mat mul1 = new Mat(new Size(64, 64),
							CvType.CV_64FC2, Scalar.all(.91));
					Mat mul2 = new Mat(new Size(64, 64),
							CvType.CV_64FC2, Scalar.all(.1));
					double[] zero = new double[1];
					zero[0] = 0;
					Mat webcam_image = new Mat(480, 640, CvType.CV_8UC3);
					Mat webcam_image_color = new Mat(480, 640, CvType.CV_8UC3);
					Names[0] = "Jacob";
					Names[1] = "Ryan";
					Names[2] = "David";
					Names[3] = "Zac";
					Names[4] = "Taran";
					Names[5] = "Jack";
					//Scanner reader = new Scanner(new File(getAssets().open(String.format("gal.txt")))); 
					Gallery = SampleApplication.getgal();
					
					
					
					
					
					while (true) {
						
						Thread.currentThread().setName(orgName + " - TRACKING" + "Frame: " + frames);
						long startTime = System.currentTimeMillis( );
						running = true;
						//System.out.println(frames);
						
						Bitmap frameBitmap = null;

						BitmapFactory.Options factoryOptions = new BitmapFactory.Options();
						factoryOptions.inSampleSize = 1;
						
						//Frame TEST
						
							
						if (Img != null) {
							byte[] jpegData = null;
							try {
								jpegData = mJpegQueue.take();
							} catch (InterruptedException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
							frameBitmap = BitmapFactory.decodeByteArray(
									jpegData, 0, jpegData.length,
									factoryOptions);
							if (frameBitmap != null)
								Utils.bitmapToMat(frameBitmap, webcam_image_color);
						}
						
						if (!webcam_image.empty()) {
							
							Imgproc.cvtColor(webcam_image_color, webcam_image,
									Imgproc.COLOR_BGR2GRAY);
							Imgproc.equalizeHist(webcam_image, webcam_image);
							if(Core.minMaxLoc(webcam_image).maxVal == 0){
								continue;
							}
							//Imgproc.resize(webcam_image, webcam_image, new Size(320, 240));
							//Imgproc.resize(webcam_image_color, webcam_image_color, new Size(320, 240));
							if(!found && frames > framecollect){
								Repaint(webcam_image_color);
							}
							
							if(frames % 150 == 0 && frames != 0){
								org.opencv.core.Rect Local = SampleApplication.FD(webcam_image);
								R = Local.y + Local.height/2;
								C = Local.x + Local.width/2;
								if(R == 0 && C == 0) continue;
								
								if(R <=32){
									R = 35;
									missedframes++;
									continue;
								}
								if(R>=webcam_image.rows()-32){
									R = webcam_image.rows()-35;
									missedframes++;
									continue;
								}
								
								if(C <=32){
									C = 35;
									missedframes++;
									continue;
								}
								if(C>=webcam_image.cols()-32){
									C = webcam_image.cols()-35;
									missedframes++;
									continue;
								}
								
								
								
								FiltLearn = cosine_window(img_norm(log_img(webcam_image.submat(R - 32, R + 32, C-32, C+32))), 64, 64, 350);	
								//FiltLearn = webcam_image.submat(R - 30, R + 30, C-30, C+30);
								ToP[0] = FiltLearn;
								Mat[] Y = PWD(ToP, 1);
								Mat mul11 = new Mat(filt_freq.size(), CvType.CV_64FC2, Scalar.all(.5));
								Mat mul21 = new Mat(filt_freq.size(), CvType.CV_64FC2, Scalar.all(.5));
								filt_freq = filt_freq.mul(mul11);
								Core.add(filt_freq, Y[0].mul(mul21),
										filt_freq);
								Core.rectangle(webcam_image_color, new Point(
										C + 33, R - 33), new Point(
										C - 33, R + 33),
										new Scalar(150));	
								
							}
							
							
							
							//Log.i("img", "" + webcam_image.dump());
							if(missedframes > 5 && frameupdate < 50){
								if(frameupdate %5 == 0){
									recon = false;
									org.opencv.core.Rect Local = SampleApplication.FD(webcam_image);
									R = Local.y + Local.height/2;
									C = Local.x + Local.width/2;
									if(R == 0 && C == 0) continue;
									
									if(R <=32){
										R = 35;
										missedframes++;
										continue;
									}
									if(R>=webcam_image.rows()-32){
										R = webcam_image.rows()-35;
										missedframes++;
										continue;
									}
									
									if(C <=32){
										C = 35;
										missedframes++;
										continue;
									}
									if(C>=webcam_image.cols()-32){
										C = webcam_image.cols()-35;
										missedframes++;
										continue;
									}
									
									
								}
								FiltLearn = cosine_window(img_norm(log_img(webcam_image.submat(R - 32, R + 32, C-32, C+32))), 64, 64, 80);	
								//FiltLearn = webcam_image.submat(R - 30, R + 30, C-30, C+30);
								ToP[0] = FiltLearn;
								Mat[] Y = PWD(ToP, 1);
								
								filt_freq = filt_freq.mul(mul1);
								Core.add(filt_freq, Y[0].mul(mul2),
										filt_freq);
								frameupdate++;
								Core.rectangle(webcam_image_color, new Point(
										C + 31, R - 31), new Point(
										C - 31, R + 31),
										new Scalar(150, 150, 150));
								Repaint(webcam_image_color);
								if(frameupdate == 10){
									frameupdate = 0;
									missedframes = 0;
								}
								continue;
							}
							
							long taskTimeMs1  = System.currentTimeMillis( ) - startTime;
							Log.i("Time", "Frame Capture: " + taskTimeMs1);
							if (frames == 0 || (frames %3 == 0 && frames < framecollect)) {
								
								 //FACE DETECTION
								org.opencv.core.Rect Local = SampleApplication.FD(webcam_image);
								lx = Local.y + Local.height/2;
								ly = Local.x + Local.width/2;
								if(lx == 0 && ly == 0) continue;
								
								if(lx <=32){
									lx = 35;
									missedframes++;
									continue;
								}
								if(lx>=webcam_image.rows()-32){
									lx = webcam_image.rows()-35;
									missedframes++;
									continue;
								}
								
								if(ly <=32){
									ly = 35;
									missedframes++;
									continue;
								}
								if(ly>=webcam_image.cols()-32){
									ly = webcam_image.cols()-35;
									missedframes++;
									continue;
								}
								
								
								
								
								
								Training[frames] = cosine_window(img_norm(log_img(webcam_image.submat(lx - 32, lx + 32, ly-32, ly+32))), 64, 64, 80);
								//Training[0] = webcam_image.submat(lx - 30, lx + 30, ly-30, ly+30);
								frames++;
							} else if (frames < framecollect) {
								
								Training[frames] = cosine_window(img_norm(log_img(webcam_image.submat(lx - 32, lx + 32, ly-32, ly+32))), 64, 64, 80);
								//Training[frames] = webcam_image.submat(lx - 30, lx + 30, ly-30, ly+30);
								frames++;
								Repaint(webcam_image_color);
								if (frames == framecollect) {
									long startTimeMs = System.currentTimeMillis( );
									Mat[] Y = PWD(Training, frames);
									long taskTimeMs  = System.currentTimeMillis( ) - startTimeMs;
									Log.i("Time", "Filter build: " + taskTimeMs);
									
									
									
									
									channelz.clear();
									for (int x = 0; x < frames; x++) {
										Core.split(Y[x], channelz);
										if (x == 0) {
											img = channelz.get(1);
											real = channelz.get(0);
										} else {
											Core.add(real, channelz.get(0),
													real);
											Core.add(img, channelz.get(1), img);
										}
									}
									channelz.set(0, real);
									channelz.set(1, img);
									Core.merge(channelz, avg);
									Mat scale3 = new Mat(avg.size(),
											CvType.CV_64FC2,
											Scalar.all(1.0 / frames));
									avg = avg.mul(scale3); // Mean calculation
									avg = avg.mul(scale3); // Divide by number
															// of
															// images
									filt_freq = avg;
									
								}

							} else { // Frames > 100
								if (frames > (framecollect + 1)) {
									
									frames++;
									
									
									if(R <=32){
										R = 35;
										missedframes++;
										continue;
									}
									if(R>=webcam_image.rows()-32){
										R = webcam_image.rows()-35;
										missedframes++;
										continue;
									}
									
									if(C <=32){
										C = 35;
										missedframes++;
										continue;
									}
									if(C>=webcam_image.cols()-32){
										C = webcam_image.cols()-35;
										missedframes++;
										continue;
									}
									
									
									
									
									FiltLearn = cosine_window(img_norm(log_img(webcam_image.submat(R - 32, R + 32, C-32, C+32))), 64, 64, 80);
									//FiltLearn = webcam_image.submat(R - 30, R + 30, C-30, C+30);
									if (found) {
										Core.rectangle(webcam_image_color, new Point(
												C + 31, R - 31), new Point(
												C - 31, R + 31),
												new Scalar(0, 0, 225));
										Core.rectangle(webcam_image_color, new Point(
												C - 1, R - 1), new Point(C + 1,
												R + 1), new Scalar(0, 0, 225));
									} else {
										org.opencv.core.Rect Local = SampleApplication.FD(webcam_image);
										if(Local.x == 0 && Local.y == 0) continue;
										R = Local.y + Local.height/2;
										C = Local.x + Local.width/2;
										
										Core.rectangle(webcam_image_color, new Point(
												C + 31, R + 31), new Point(
												C - 31, R - 31), new Scalar(0, 0, 0));
										
									}
									// REPAINT
									int fd = -1;
									if (found) {
										if (!recon || frames % 40 == 0) {
											fd = FD(webcam_image.submat(R - 40, R + 40,
													C - 40, C + 40));
											if (fd != -1) {
												recon = true;
												person = fd;
											}
										}
										if (fd != -1 || recon) {
											Core.putText(webcam_image_color, Names[person],
													new Point(C + 32, R - 10), 0, 1,
													new Scalar(225));
										}
									}
									Repaint(webcam_image_color);
									
								} else if (frames == framecollect) {
									frames += 10;
									R = lx;
									C = ly;
									FiltLearn = cosine_window(img_norm(log_img(webcam_image.submat(lx - 32, lx + 32, ly-32, ly+32))), 64, 64, 80);
									//FiltLearn = webcam_image.submat(lx - 30, lx + 30, ly-30, ly+30);
								}
								ToP[0] = FiltLearn;
								long startTimeMs = System.currentTimeMillis( );
								Mat[] Y = PWD(ToP, 1);
								long taskTimeMs  = System.currentTimeMillis( ) - startTimeMs;
								// Correlating the Image with the filter, and
								// returning the result
								

								startTimeMs = System.currentTimeMillis( );
								
								Mat corrplane = Correlation(FiltLearn, filt_freq);
								taskTimeMs  = System.currentTimeMillis( ) - startTimeMs;
								Log.i("Time", "Correlation: " + taskTimeMs);
								
								
								
								//System.out.println("CORRELATION: " + duration);
								startTimeMs = System.currentTimeMillis( );
								// Sending the new image to PWD
								double value = 0;
								int locx = 0, locy = 0;
								double sum = 0;
								// Locating the peak, and storing its value
								MinMaxLocResult result = null;
								result = Core.minMaxLoc(corrplane);
								value = result.maxVal;
								locx = (int) result.maxLoc.x;
								locy = (int) result.maxLoc.y;
								/*for (int x = 0; x < corrplane.cols(); x++) {
									for (int y = 0; y < corrplane.rows(); y++) {
										if (corrplane.get(y, x)[0] > value) {
											value = corrplane.get(y, x)[0];
											locx = x;
											locy = y;
										}
									}
								}
								*/
								System.out.println("Peak:" + locx + ", " + locy + " Val: " + value);
								//System.out.println("CHECK ------ 4" + locx + " " + locy);
								//Mat mask_one = new Mat(corrplane.size(),
								//		CvType.CV_64F, Scalar.all(1.0));
								
								// Placing the zeros in the mask
								for (int x = locx - 5; x < locx + 6; x++) {
									for (int y = locy - 5; y < locy + 6; y++) {
										corrplane.put(x, y, zero);
									}
								}
								//System.out.println("CHECK ------ 5");
								// multiplying by the mask
								// Below are the PSR calculations
								//corrplane = corrplane.mul(mask_one);
								sum = 0;
								
								//Log.i("Cor", frames + corrplane.dump());
								double[] buf3 = new double[64*64];
								corrplane.get(0, 0, buf3);
								
								for (int x = 0; x < corrplane.total(); x++) {
										sum += buf3[x];
								}
	
								sum = sum / (corrplane.rows() * corrplane.cols() - 5 * 6);
								MatOfDouble mean = new MatOfDouble();
								MatOfDouble stddev = new MatOfDouble();
								Core.meanStdDev(corrplane, mean, stddev);
								double corrStddev = stddev.get(0, 0)[0];
								PSR = (value - sum) / corrStddev;
								taskTimeMs  = System.currentTimeMillis( ) - startTimeMs;
								Log.i("Time", "PSR Calculation: " + taskTimeMs);
								System.out.println("PSR: " + PSR);
								if (PSR > 5) {
									found = true;
									R = locy + (R - 32);
									C = locx + (C - 32);
									Log.i("RC", "R: "+ R + " C: " + C);
									
									if(R <=32){
										R = 35;
										missedframes++;
										continue;
									}
									if(R>=webcam_image.rows()-32){
										R = webcam_image.rows()-35;
										missedframes++;
										continue;
									}
									
									if(C <=32){
										C = 35;
										missedframes++;
										continue;
									}
									if(C>=webcam_image.cols()-32){
										C = webcam_image.cols()-35;
										missedframes++;
										continue;
									}
									
									// Learning rate from input frame
									/*mul1 = new Mat(filt_freq.size(),
											CvType.CV_64FC2, Scalar.all(.825));
									mul2 = new Mat(filt_freq.size(),
											CvType.CV_64FC2, Scalar.all(.175));
											*/
									filt_freq = filt_freq.mul(mul1);
									Core.add(filt_freq, Y[0].mul(mul2),
											filt_freq);
									missedframes = 0;
								} else {
									found = false;
									missedframes++;
									
								}

							}// End of else (frames > 100)
						}
						long taskTime  = System.currentTimeMillis( ) - startTime;
						Log.i("FPS", "CURRENT FPS: " + 1000.0/taskTime);
					}// End of loop
				}
			}
			
			public Mat[] PWD(Mat entry[], int img_nums) {
				ArrayList<Mat> channel = new ArrayList<Mat>();
				int imgsize = entry[0].rows() * entry[0].cols();
				double alpha = .001;
				double beta = 1 - alpha;
				int num_img = img_nums;
				Mat[] Y = new Mat[num_img];
				Mat[] complexI = new Mat[num_img];
				Mat scale11 = new Mat(entry[0].size(), CvType.CV_64FC2,
						Scalar.all(1.0 / Math.sqrt(entry[0].cols()
								* entry[0].rows())));

				for (int x = 0; x < num_img; x++) {
					complexI[x] = new Mat(entry[x].size(), CvType.CV_64FC2);
					entry[x].convertTo(entry[x], CvType.CV_64FC2);
					Core.dft(entry[x], complexI[x], Core.DFT_COMPLEX_OUTPUT, 0);
					complexI[x] = complexI[x].mul(scale11);
				}
				Mat F = new Mat();
				F = Meancalc(complexI, num_img);
				Scalar s = new Scalar(imgsize);
				Scalar s1 = new Scalar(1.0 / num_img);
				Mat scale1 = new Mat(F.size(), CvType.CV_64F,
						s1.all(1.0 / num_img));
				Mat scale = new Mat(F.size(), CvType.CV_64F, s.all(imgsize));

				F = F.mul(scale1);
				F = F.mul(scale);
				F = normalize(F);
				Mat scale2 = new Mat(F.size(), CvType.CV_64F, s.all(beta));
				F = F.mul(scale2);
				Core.add(F, s.all(alpha), F);
				normalize(F);
				for (int x = 0; x < num_img; x++) {
					Mat Real = new Mat(F.size(), 0);
					Mat Img = new Mat(F.size(), 0);
					Y[x] = complexI[x];
					Core.split(Y[x], channel);
					channel.get(1).put(0, 0, 1);
					Core.divide(channel.get(0), F, Real);
					Core.divide(channel.get(1), F, Img);

					channel.set(0, Real);
					channel.set(1, Img);
					Core.merge(channel, Y[x]);
					Y[x].put(0, 0, 0, 0);
				}
				return Y;
			}

			public Mat normalize(Mat entry) {
				MinMaxLocResult result = null;
				result = Core.minMaxLoc(entry);
				double max = 0;
				/*
				 * for (int x = 0; x < entry.cols(); x++) { for (int y = 0; y <
				 * entry.rows(); y++) { if (entry.get(y, x)[0] > max) { max =
				 * entry.get(y, x)[0]; } } }
				 */

				Scalar s = new Scalar(1.0 / result.maxVal);
				Mat scale = new Mat(entry.size(), CvType.CV_64F, s.all(1.0 / result.maxVal));
				entry = entry.mul(scale);
				return entry;
			}
				ArrayList<Mat> channel1 = new ArrayList<Mat>();
			public Mat Meancalc(Mat entry[], int numimg) {
				channel1.clear();
				Mat Fin = null;
				Mat abs = null;
				Mat Real = null;
				Mat Img = null;
				for (int x = 0; x < numimg; x++) {
					if (x == 0) {
						Core.split(entry[x], channel1);
						Real = channel1.get(0);
						Img = channel1.get(1);
						abs = Real.mul(Real);
						Core.add(abs, Img.mul(Img), abs);
						Core.sqrt(abs, abs);
						Fin = abs.mul(abs);
					} else {
						Core.split(entry[x], channel1);
						Real = channel1.get(0);
						Img = channel1.get(1);
						abs = Real.mul(Real);
						Core.add(abs, Img.mul(Img), abs);
						Core.sqrt(abs, abs);
						Core.add(Fin, abs.mul(abs), Fin);
					}
				}
				return Fin;
			}

			
			double[] buf = new double[64*64*2];
			double[] imgbuf = new double[64*64*2];
			double[] filtbuf = new double[64*64*2];
			public Mat Correlation(Mat img2corr, Mat Ffilt) {
				Mat correl = new Mat(Ffilt.size(), Ffilt.type());
				Mat imgdft = img2corr;
				imgdft.convertTo(imgdft, CvType.CV_64FC2);
				Core.dft(img2corr, imgdft, Core.DFT_COMPLEX_OUTPUT, 0);
				ArrayList<Mat> channels = new ArrayList<Mat>();
				//conj
				Core.split(Ffilt, channels);
				Mat scale5 = new Mat(imgdft.size(), CvType.CV_64F, Scalar.all(-1));
				Mat Fimg = channels.get(1);
				Fimg = Fimg.mul(scale5);
				channels.set(1, Fimg);


				Core.merge(channels, Ffilt);
				//writer.println(Ffilt.dump());
				Mat scale3 = new Mat(imgdft.size(), CvType.CV_64FC2, Scalar.all(1.0 / (imgdft.rows()*imgdft.cols())));
				imgdft = imgdft.mul(scale3);
				//writer.println(imgdft.dump());



				//writer.println(Ffilt.dump());

				//correl = Ffilt.mul(imgdft);
				
				
				img2corr.get(0, 0, imgbuf);
				Ffilt.get(0, 0, filtbuf);
				//for(int x = 0; x < imgdft.cols(); x++){
					//for(int y=0; y<imgdft.rows(); y++){
				for(int x = 0; x < imgdft.rows()*imgdft.cols()*2; x = x+2){
					
						//double Ire = imgdft.get(x, y)[0];
						//double Iim = imgdft.get(x, y)[1];
						//double Fre= Ffilt.get(x, y)[0];
						//double Fim = Ffilt.get(x, y)[1];

						//double[] result = new double[2];

						//result[0] = Ire*Fre - Iim * Fim;
						//result[1] = Fre * Iim + Ire * Fim;
						//correl.put(x, y, result);
						
						buf[x] = imgbuf[x]*filtbuf[x] - imgbuf[x+1]*filtbuf[x+1];
						buf[x+1] = filtbuf[x]*imgbuf[x+1] + imgbuf[x]*filtbuf[x+1];
					
				//}
					}
				correl.put(0, 0, buf);
				Mat fin = imgdft;
				Core.dft(correl, fin, Core.DFT_INVERSE, 0);



				ArrayList<Mat> channel = new ArrayList<Mat>();
				Core.split(fin, channel);
				Mat corr1 = channel.get(0);
				Mat scale4 = new Mat(fin.size(), CvType.CV_64F, Scalar.all((imgdft.rows()*imgdft.cols())));
				//corr1 = corr1.mul(scale4);
				int cx = corr1.cols()/2;
				int cy = corr1.rows()/2;

				Mat q0 = new Mat(corr1, new org.opencv.core.Rect(0,0,cx,cy));
				Mat q1 = new Mat(corr1, new org.opencv.core.Rect(cx,0,cx,cy));
				Mat q2 = new Mat(corr1, new org.opencv.core.Rect(0,cy,cx,cy));
				Mat q3 = new Mat(corr1, new org.opencv.core.Rect(cx,cy,cx,cy));

				Mat tmp = new Mat();
				q0.copyTo(tmp);
				q3.copyTo(q0);
				tmp.copyTo(q3);

				q1.copyTo(tmp);
				q2.copyTo(q1);
				tmp.copyTo(q2);

				return corr1;
			}

			public void Repaint(Mat in) {
				if (Img != null) {
					BitmapFactory.Options factoryOptions = new BitmapFactory.Options();
					factoryOptions.inSampleSize = 1;
					//Imgproc.resize(in, in, new Size(640, 480));
					Bitmap frameBitmap = BitmapFactory.decodeByteArray(Img, 0,
							Img.length, factoryOptions);
					Utils.matToBitmap(in, frameBitmap);

					if (mInMutableAvailable) {
						initInBitmap(factoryOptions);
					}
					if (mInMutableAvailable) {
						setInBitmap(factoryOptions, frameBitmap);
					}
					drawFrame(frameBitmap);
				}
			}
			
			
			public Mat log_img(Mat img){
				//double[] buf4 = new double[(int) (img.total())];
				//double[] buf5 = new double[(int) (img.total())];
				
				/*img.convertTo(img, CvType.CV_64F);
				Mat nimg = new Mat(img.size(), CvType.CV_64F);
				img.get(0, 0, normbuf);
				for(int x = 0; x < img.total(); x++){					
					normbuf2[x] = 0.3 * Math.log10(1+normbuf[x]);	
				}
				nimg.put(0,0,normbuf2);
				*/
				return img;
			}
			
			public Mat cosine_window(Mat img, int x, int y, int vrns){
				return img.mul(gi);
			}
			
			public int FD(Mat in) {
				double PSRM = 0;
				int ploc = -1;
				for (int y = 0; y < G_size-1; y++) {
					Mat complexI = new Mat();
					in.convertTo(complexI, CvType.CV_64FC2);
					Core.dft(complexI, complexI, Core.DFT_COMPLEX_OUTPUT, 0);

					double[] buf = new double[80 * 80 * 2];
					double[] imgbuf = new double[80 * 80 * 2];
					double[] filtbuf = new double[80 * 80 * 2];

					complexI.get(0, 0, imgbuf);
					Gallery[y].get(0, 0, filtbuf);
					for (int x = 0; x < complexI.rows() * complexI.cols() * 2; x = x + 2) {
						buf[x] = imgbuf[x] * filtbuf[x] - imgbuf[x + 1]
								* filtbuf[x + 1];
						buf[x + 1] = filtbuf[x] * imgbuf[x + 1] + imgbuf[x]
								* filtbuf[x + 1];
					}
					complexI.put(0, 0, buf);
					Core.dft(complexI, complexI, Core.DFT_COMPLEX_OUTPUT, 0);

					ArrayList<Mat> channel = new ArrayList<Mat>();
					Core.split(complexI, channel);
					Mat corr1 = channel.get(0);

					int cx = corr1.cols() / 2;
					int cy = corr1.rows() / 2;

					Mat q0 = new Mat(corr1, new org.opencv.core.Rect(0, 0, cx, cy));
					Mat q1 = new Mat(corr1, new org.opencv.core.Rect(cx, 0, cx, cy));
					Mat q2 = new Mat(corr1, new org.opencv.core.Rect(0, cy, cx, cy));
					Mat q3 = new Mat(corr1, new org.opencv.core.Rect(cx, cy, cx, cy));

					Mat tmp = new Mat();
					q0.copyTo(tmp);
					q3.copyTo(q0);
					tmp.copyTo(q3);

					q1.copyTo(tmp);
					q2.copyTo(q1);
					tmp.copyTo(q2);

					MinMaxLocResult result = null;
					result = Core.minMaxLoc(corr1);
					double value = result.maxVal;
					int locx = (int) result.maxLoc.x;
					int locy = (int) result.maxLoc.y;

					// System.out.println(locx + " " + locy);

					/*double[] zero = new double[1];
					zero[0] = 0;
					if (locx > 70 || locx < 10 || locy > 70 || locy < 10) {
						continue;
					}
					for (int x = locx - 2; x < locx + 3; x++) {
						for (int yy = locy - 2; yy < locy + 3; yy++) {
							corr1.put(x, yy, zero);
						}
					}

					if (locx > 70 || locx < 10 || locy > 70 || locy < 10) {
						continue;
					}

					Mat fd_result = corr1.submat(locx - 10, locx + 10, locy - 10,
							locy + 10);

					MatOfDouble mean = new MatOfDouble();
					MatOfDouble stddev = new MatOfDouble();
					Core.meanStdDev(fd_result, mean, stddev);
					double[] buf3 = new double[80 * 80*2];
					corr1.get(0, 0, buf3);
					double sum = 0;
					for (int x = 0; x < corr1.total(); x++) {
						sum += buf3[x];
					}
					double meanr = sum / (corr1.total() - 5*5);
					double corrStddev = stddev.get(0, 0)[0];
					double PSR = (value - meanr) / corrStddev;
					//System.out.println(PSR + "  <---- FD");
					 * 
					 */
					double[] sidelobeRegion = new double[544];
					for(int ia = 0; ia!=544; ia++){
						sidelobeRegion[ia] = 0;
					}
					
					int m = (int) (complexI.size().width - 1);
					int pos = 0;
					
					int RlowBd = (locx-12)>0?(locx-12):0;	// max(R-12, 1); 
					int ClowBd = (locy-12)>0?(locy-12):0;	// max(C-12, 1); 
					int RhighBd = (locx+12)<m?(locx+12):m; // min(R+12, m); 
					int ChighBd = (locy+12)<m?(locy+12):m; // min(C+12, m); 
					int RleftBd1 = (locx-4)>0?(locx-4):0; // max(R-4,1); 
					int RleftBd2 = (locx-5)>0?(locx-5):0; // max(R-5,1); 
					int RrightBd1 = (locx+4)<m?(locx+4):m; // min(R+4,m); 
					int RrightBd2 = (locx+5)<m?(locx+5):m; // min(R+5,m); 
					int CleftBd2 = (locy-5)>0?(locy-5):0; // max(C-5,1); 
					int CrightBd2 = (locy+5)<m?(locy+5):m; // min(C+5,m); 
					
					for(int ia=RlowBd; ia<=RleftBd2; ++ia){ 
						for(int ib=ClowBd; ib<=ChighBd; ++ib){ 
							sidelobeRegion[pos] = corr1.get(ia,ib)[0]; 
							++pos;
						}
					}
					
					// Excluding the central mask 
					for(int ia=RleftBd1; ia<=RrightBd1; ++ia){ 
						for(int ib=ClowBd; ib<=CleftBd2; ++ib){ 
							sidelobeRegion[pos] = corr1.get(ia,ib)[0]; 
							++pos;
						}
						for(int ib=CrightBd2; ib<=ChighBd; ++ib){ 
							sidelobeRegion[pos] = corr1.get(ia,ib)[0]; 
							++pos;
						}
					}
					
					for(int ia=RrightBd2; ia<=RhighBd; ++ia){ 
						for(int ib=ClowBd; ib<=ChighBd; ++ib){ 
							sidelobeRegion[pos] = corr1.get(ia,ib)[0]; 
							++pos;
						}
					}

					
					double mean=0;	// Calculate the mean of the sidelobe region 
					for(int ia=0; ia!=pos; ++ia){ 
						mean += sidelobeRegion[ia]; 
					}
					mean = mean/544;

					double sigma=0;	// Calculate the standard deviation of the sidelobe region 
					for(int ia=0; ia!=544; ++ia) {
						sigma += (sidelobeRegion[ia]-mean)*(sidelobeRegion[ia]-mean); 
					}
					sigma = sigma/544;
					
					sigma = Math.sqrt(sigma);
					double PSR = (value-mean)/sigma;
					//System.out.println(PSR + " " + y);
					if (PSR > 8) {
						PSRM = PSR;
						ploc = y;
						return y;
					}
				}
				return ploc;
			}
			
			public Mat img_norm(Mat img){
				//double[] buf4 = new double[(int) (img.total())];
				img.convertTo(img, CvType.CV_64F);
				Scalar s1 = new Scalar(0);
				s1 = Core.mean(img);
				
				Core.subtract(img, s1, img);
				
				Mat Fin = img.mul(img);

				/*double sum1 = 0;
				for(int x = 0; x < Fin.cols(); x++){
					for(int y = 0; y < Fin.rows(); y++){
						sum1 += Fin.get(y, x)[0];						
					}
				}
				*/
				Fin.get(0, 0, buf4);
				double sum1 = 0;
				for(int x = 0; x < Fin.total(); x++){				
						sum1 += buf4[x];						
				}
				
				
				sum1 = Math.sqrt(sum1);
				
				Mat div = new Mat(Fin.size(), CvType.CV_64F, Scalar.all(1.0/sum1) );
				
				Fin = img.mul(div);
				
				//writer.println(div.dump());
				//Fin = Fin.mul(div);
				
				
				return Fin;
			}
		}.start();
		return true;
	}

	/**
	 * Request to stop retrieving and drawing liveview data.
	 */
	public void stop() {
		mWhileFetching = false;
	}

	/**
	 * Check to see whether start() is already called.
	 * 
	 * @return true if start() is already called, false otherwise.
	 */
	public boolean isStarted() {
		return mWhileFetching;
	}

	@TargetApi(Build.VERSION_CODES.HONEYCOMB)
	private void initInBitmap(BitmapFactory.Options options) {
		options.inBitmap = null;
		options.inMutable = true;
	}

	@TargetApi(Build.VERSION_CODES.HONEYCOMB)
	private void clearInBitmap(BitmapFactory.Options options) {
		if (options.inBitmap != null) {
			options.inBitmap.recycle();
			options.inBitmap = null;
		}
	}

	@TargetApi(Build.VERSION_CODES.HONEYCOMB)
	private void setInBitmap(BitmapFactory.Options options, Bitmap bitmap) {
		options.inBitmap = bitmap;
	}

	// Draw frame bitmap onto a canvas.
	private void drawFrame(Bitmap frame) {
		/*
		 * if (frame.getWidth() != mPreviousWidth || frame.getHeight() !=
		 * mPreviousHeight) { onDetectedFrameSizeChanged(frame.getWidth(),
		 * frame.getHeight()); return; }
		 */
		Canvas canvas = getHolder().lockCanvas();
		if (canvas == null) {
			return;
		}

		int w = frame.getWidth();
		int h = frame.getHeight();

		Rect src = new Rect(0, 0, w, h);

		float by = Math.min((float) getWidth() / w, (float) getHeight() / h);
		int offsetX = (getWidth() - (int) (w * by)) / 2;
		int offsetY = (getHeight() - (int) (h * by)) / 2;
		Rect dst = new Rect(offsetX, offsetY, getWidth() - offsetX, getHeight()
				- offsetY);
		canvas.drawBitmap(frame, src, dst, mFramePaint);
		getHolder().unlockCanvasAndPost(canvas);
	}

	// Called when the width or height of liveview frame image is changed.
	private void onDetectedFrameSizeChanged(int width, int height) {
		Log.d(TAG, "Change of aspect ratio detected");
		mPreviousWidth = width;
		mPreviousHeight = height;
		drawBlackFrame();
		drawBlackFrame();
		drawBlackFrame(); // delete triple buffers
	}

	// Draw black screen.
	private void drawBlackFrame() {
		Canvas canvas = getHolder().lockCanvas();
		if (canvas == null) {
			return;
		}

		Paint paint = new Paint();
		paint.setColor(Color.BLACK);
		paint.setStyle(Paint.Style.FILL);

		canvas.drawRect(new Rect(0, 0, getWidth(), getHeight()), paint);
		getHolder().unlockCanvasAndPost(canvas);
	}

	// Parse JSON and returns a error code.
	private static boolean isErrorReply(JSONObject replyJson) {
		boolean hasError = (replyJson != null && replyJson.has("error"));
		return hasError;
	}

}