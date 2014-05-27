/*
 * Copyright 2013 Sony Corporation
 */

package com.example.sony.cameraremote;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;

import android.app.Application;
import android.content.Context;
import android.util.Log;

/**
 * A Application class for the sample application.
 *
 */


public class SampleApplication extends Application {
	protected static final String TAG = "Cheese";
	private File                   mCascadeFile;
	public static  CascadeClassifier      mJavaDetector;
	public static Mat[] Gallery = new Mat[6];


	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
	    @Override
	    public void onManagerConnected(int status) {
	            switch (status) {
	                case LoaderCallbackInterface.SUCCESS:
	                {
	                    Log.i(TAG, "OpenCV loaded successfully");

	                    // Load native library after(!) OpenCV initialization
	                   

	                    try {
	                        // load cascade file from application resources
	                    	Scanner scan = new Scanner(getResources().openRawResource(R.raw.newgal));
	                        Mat hold = new Mat(new Size(80, 80), CvType.CV_64FC2, new Scalar(0, 0));
	                		Mat hold1 = new Mat(new Size(80, 80), CvType.CV_64FC2, new Scalar(0, 0));
	                		Mat hold2 = new Mat(new Size(80, 80), CvType.CV_64FC2, new Scalar(0, 0));
	                		Mat hold3 = new Mat(new Size(80, 80), CvType.CV_64FC2, new Scalar(0, 0));
	                		Mat hold4 = new Mat(new Size(80, 80), CvType.CV_64FC2, new Scalar(0, 0));
	                		Gallery[0] = hold;
	                		Gallery[1] = hold1;
	                		Gallery[2] = hold2;
	                		Gallery[3] = hold3;
	                		Gallery[4] = hold4;
	                		double buff[] = new double[80 * 80 * 2];
	                		double buff1[] = new double[80 * 80 * 2];
	                		double buff2[] = new double[80 * 80 * 2];
	                		double buff3[] = new double[80 * 80 * 2];
	                		double buff4[] = new double[80 * 80 * 2];
	                		
	                		for (int x = 0; x < 80 * 80 * 2; x++) {
	                			buff[x] = scan.nextDouble();
	                		}
	                		Gallery[0].put(0, 0, buff);

	                		// Scanner scan1 = new Scanner(new File("person2.txt"));
	                		for (int x = 0; x < 80 * 80 * 2; x++) {
	                			buff1[x] = scan.nextDouble();
	                		}
	                		// scan1.close();
	                		Gallery[1].put(0, 0, buff1);

	                		// Scanner scan2 = new Scanner(new File("person3.txt"));
	                		for (int x = 0; x < 80 * 80 * 2; x++) {
	                			buff2[x] = scan.nextDouble();
	                		}
	                		// scan2.close();
	                		Gallery[2].put(0, 0, buff2);
	                		
	                		for (int x = 0; x < 80 * 80 * 2; x++) {
	                			buff3[x] = scan.nextDouble();
	                		}
	                		// scan2.close();
	                		Gallery[3].put(0, 0, buff3);
	                		
	                		for (int x = 0; x < 80 * 80 * 2; x++) {
	                			buff4[x] = scan.nextDouble();
	                		}
	                		
	                		Gallery[4].put(0, 0, buff4);
	                		scan.close();
	                    	
	                    	
	                    	
	                        InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
	                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
	                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
	                        FileOutputStream os = new FileOutputStream(mCascadeFile);
	                        
	                        
	                        byte[] buffer = new byte[4096];
	                        int bytesRead;
	                        while ((bytesRead = is.read(buffer)) != -1) {
	                            os.write(buffer, 0, bytesRead);
	                        }
	                        is.close();
	                        os.close();

	                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
	                        if (mJavaDetector.empty()) {
	                            Log.e(TAG, "Failed to load cascade classifier");
	                            mJavaDetector = null;
	                        } else
	                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

	                        cascadeDir.delete();

	                    } catch (IOException e) {
	                        e.printStackTrace();
	                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
	                    }
	                } break;
	                default:
	                {
	                    super.onManagerConnected(status);
	                } break;
	            }
	        }
	};

    
	
	
    private ServerDevice mTargetDevice;

    /**
     * Sets a target server object to transmit to SampleCameraActivity.
     * 
     * @param device
     */
    public void setTargetServerDevice(ServerDevice device) {
        mTargetDevice = device;
    }
    
    public static Mat[] getgal(){
    	return Gallery;
    }
    public static Rect FD (Mat in){
    	System.out.println(in.dump());
    	MatOfRect faces = new MatOfRect();
    	mJavaDetector.detectMultiScale(in, faces, 1.1, 2, 2 // TODO: objdetect.CV_HAAR_SCALE_IMAGE
    			, new Size(0, 0), new Size());
		org.opencv.core.Rect Local = new org.opencv.core.Rect();
		
		if(faces.toArray().length > 0){ //if face detected
			Local  = faces.toArray()[0]; //Coordinates of detected face -> Local
		}
		
    	return Local;
    }

    /**
     * Returns a target server object to get from SampleDeviceSearchActivity.
     * 
     * @param device
     */
    public ServerDevice getTargetServerDevice() {
        return mTargetDevice;
    }

    @Override
    public void onCreate() {
    	
        super.onCreate();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_5, this, mLoaderCallback);
    }
}
