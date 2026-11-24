package com.paigen.ai;

import android.content.Context;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.util.Base64;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class CallRecorderManager {
    private final Context context;
    private MediaRecorder recorder;
    private MediaPlayer player;
    private File currentFile;
    private long startTimeMs = 0L;
    private boolean isRecording = false;

    public CallRecorderManager(Context context) {
        this.context = context.getApplicationContext();
    }

    public File getRecordingsDir() {
        File dir = new File(context.getFilesDir(), "recordings");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        return dir;
    }

    public synchronized String startRecording() throws IOException {
        stopRecording();
        File out = new File(getRecordingsDir(), "paigen_call_" + System.currentTimeMillis() + ".m4a");
        currentFile = out;
        recorder = new MediaRecorder();
        try {
            recorder.setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION);
        } catch (Exception e) {
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
        }
        recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
        recorder.setAudioSamplingRate(44100);
        recorder.setAudioEncodingBitRate(96000);
        recorder.setOutputFile(out.getAbsolutePath());
        recorder.prepare();
        recorder.start();
        startTimeMs = System.currentTimeMillis();
        isRecording = true;
        return out.getAbsolutePath();
    }

    public static class RecordingResult {
        public final String filePath;
        public final double durationSec;

        public RecordingResult(String filePath, double durationSec) {
            this.filePath = filePath;
            this.durationSec = durationSec;
        }
    }

    public synchronized RecordingResult stopRecording() {
        String path = currentFile != null ? currentFile.getAbsolutePath() : "";
        double dur = isRecording ? (System.currentTimeMillis() - startTimeMs) / 1000.0 : 0.0;
        if (recorder != null) {
            try {
                if (isRecording) {
                    recorder.stop();
                }
            } catch (Exception ignored) {
            }
            try {
                recorder.release();
            } catch (Exception ignored) {
            }
            recorder = null;
        }
        isRecording = false;
        return new RecordingResult(path, Math.max(0.0, dur));
    }

    public boolean isRecording() {
        return isRecording;
    }

    public synchronized void playFile(String path, Runnable onDone) throws IOException {
        stopPlayback();
        player = new MediaPlayer();
        player.setDataSource(path);
        player.setOnCompletionListener(mp -> {
            stopPlayback();
            if (onDone != null) onDone.run();
        });
        player.prepare();
        player.start();
    }

    public synchronized void playMp3Base64(String base64Audio, Runnable onDone) throws IOException {
        stopPlayback();
        byte[] bytes = Base64.decode(base64Audio, Base64.DEFAULT);
        File tmp = File.createTempFile("paigen_tts_", ".mp3", context.getCacheDir());
        try (FileOutputStream fos = new FileOutputStream(tmp)) {
            fos.write(bytes);
        }
        player = new MediaPlayer();
        player.setDataSource(tmp.getAbsolutePath());
        player.setOnCompletionListener(mp -> {
            stopPlayback();
            tmp.delete();
            if (onDone != null) onDone.run();
        });
        player.prepare();
        player.start();
    }

    public synchronized void stopPlayback() {
        if (player != null) {
            try {
                if (player.isPlaying()) player.stop();
            } catch (Exception ignored) {
            }
            try {
                player.release();
            } catch (Exception ignored) {
            }
            player = null;
        }
    }

    public static String fileToBase64(String path) {
        File f = new File(path);
        if (!f.exists()) return "";
        try (FileInputStream fis = new FileInputStream(f);
             ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = fis.read(buf)) > 0) {
                bos.write(buf, 0, n);
            }
            return Base64.encodeToString(bos.toByteArray(), Base64.NO_WRAP);
        } catch (Exception e) {
            return "";
        }
    }
}
