package com.paigen.ai;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;

import java.net.HttpURLConnection;
import java.net.URL;

public class PaiGenBridgeService extends Service {
    private static final String CHANNEL_BG = "paigen_pc_link";
    private static final String CHANNEL_CALL = "paigen_call_prompt";
    private static final int NOTIF_BG_ID = 1001;
    private static final int NOTIF_CALL_ID = 2002;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private boolean running = false;
    private boolean pcOnline = false;

    private final Runnable heartbeatTask = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            new Thread(() -> {
                boolean online = pingPcServer();
                if (online != pcOnline) {
                    pcOnline = online;
                    updateBgNotification();
                }
            }).start();
            handler.postDelayed(this, 15000L);
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        ensureChannels();
        running = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIF_BG_ID, buildBgNotification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(NOTIF_BG_ID, buildBgNotification());
        }
        handler.post(heartbeatTask);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        running = false;
        handler.removeCallbacks(heartbeatTask);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private void ensureChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationManager nm = getSystemService(NotificationManager.class);
            NotificationChannel bg = new NotificationChannel(
                CHANNEL_BG, "PaiGen PC Link", NotificationManager.IMPORTANCE_LOW
            );
            NotificationChannel call = new NotificationChannel(
                CHANNEL_CALL, "PaiGen Incoming Call Prompt", NotificationManager.IMPORTANCE_HIGH
            );
            call.enableVibration(true);
            nm.createNotificationChannel(bg);
            nm.createNotificationChannel(call);
        }
    }

    private Notification buildBgNotification() {
        Intent openIntent = new Intent(this, MainActivity.class);
        PendingIntent pi = PendingIntent.getActivity(
            this, 0, openIntent, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        String status = pcOnline ? "Connected to PC Workstation • Call Shield ON"
                : "Standby on Mobile • Tap to configure PC IP";
        Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_BG)
                : new Notification.Builder(this);
        return b.setContentTitle("PaiGen AI Guardian")
                .setContentText(status)
                .setSmallIcon(R.drawable.ic_paigen_launcher)
                .setContentIntent(pi)
                .setOngoing(true)
                .build();
    }

    private void updateBgNotification() {
        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm != null) {
            nm.notify(NOTIF_BG_ID, buildBgNotification());
        }
    }

    private boolean pingPcServer() {
        try {
            SharedPreferences prefs = getSharedPreferences("paigen_prefs", MODE_PRIVATE);
            String baseUrl = prefs.getString("server_url", "http://192.168.11.146:8765");
            URL url = new URL(baseUrl + "/api/ping");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setConnectTimeout(2500);
            conn.setReadTimeout(2500);
            int code = conn.getResponseCode();
            conn.disconnect();
            return code == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public static void triggerIncomingCallPrompt(Context ctx, String callerName, String callerNumber) {
        NotificationManager nm = ctx.getSystemService(NotificationManager.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && nm != null) {
            NotificationChannel ch = new NotificationChannel(
                CHANNEL_CALL, "PaiGen Incoming Call Prompt", NotificationManager.IMPORTANCE_HIGH
            );
            nm.createNotificationChannel(ch);
        }

        Intent promptIntent = new Intent(ctx, IncomingCallPromptActivity.class);
        promptIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        promptIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NAME, callerName);
        promptIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NUMBER, callerNumber);

        PendingIntent fullScreenPi = PendingIntent.getActivity(
            ctx, 200, promptIntent, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );

        Intent talkIntent = new Intent(ctx, MainActivity.class);
        talkIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        talkIntent.putExtra("action", "paigen_take_call");
        talkIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NAME, callerName);
        talkIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NUMBER, callerNumber);
        PendingIntent talkPi = PendingIntent.getActivity(
            ctx, 201, talkIntent, PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );

        if (nm != null) {
            Notification.Builder b = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                    ? new Notification.Builder(ctx, CHANNEL_CALL)
                    : new Notification.Builder(ctx);
            Notification notif = b.setContentTitle("Incoming Call: " + callerName)
                    .setContentText("PaiGen asks: Should I talk? (" + callerNumber + ")")
                    .setSmallIcon(R.drawable.ic_paigen_launcher)
                    .setFullScreenIntent(fullScreenPi, true)
                    .setContentIntent(fullScreenPi)
                    .setAutoCancel(true)
                    .addAction(R.drawable.ic_paigen_launcher, "Let PaiGen Talk", talkPi)
                    .build();
            nm.notify(NOTIF_CALL_ID, notif);
        }

        try {
            ctx.startActivity(promptIntent);
        } catch (Exception ignored) {
        }
    }
}
