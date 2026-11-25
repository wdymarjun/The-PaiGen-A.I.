package com.paigen.ai;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

public class PaiGenOverlayService extends Service {
    public static final String ACTION_SHOW = "com.paigen.ai.OVERLAY_SHOW";
    public static final String ACTION_UPDATE = "com.paigen.ai.OVERLAY_UPDATE";
    public static final String ACTION_HIDE = "com.paigen.ai.OVERLAY_HIDE";

    private static volatile boolean isRunning = false;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private WindowManager windowManager;
    private WindowManager.LayoutParams params;
    private FrameLayout rootContainer;

    private LinearLayout bubbleView;
    private TextView bubbleSubLabel;

    private LinearLayout hudView;
    private TextView txtOverlayStatus;
    private TextView txtOverlayCaller;
    private TextView txtOverlayTranscript;

    private boolean expanded = false;
    private String currentCallerName = "PaiGen Active";
    private String currentCallerNumber = "";
    private String currentStatus = "STANDBY • READY";
    private String currentTranscript = "Floating over apps. Tap 'Let PaiGen Talk' to handle calls hands-free.";

    public static boolean isOverlayActive() {
        return isRunning;
    }

    public static void show(Context ctx, boolean expand, String callerName, String callerNumber, String statusText) {
        if (!Settings.canDrawOverlays(ctx)) return;
        Intent i = new Intent(ctx, PaiGenOverlayService.class);
        i.setAction(ACTION_SHOW);
        i.putExtra("expand", expand);
        if (callerName != null) i.putExtra("caller_name", callerName);
        if (callerNumber != null) i.putExtra("caller_number", callerNumber);
        if (statusText != null) i.putExtra("status_text", statusText);
        try {
            ctx.startService(i);
        } catch (Exception ignored) {
        }
    }

    public static void updateStatus(Context ctx, String statusText, String transcriptText) {
        if (!isRunning && !Settings.canDrawOverlays(ctx)) return;
        Intent i = new Intent(ctx, PaiGenOverlayService.class);
        i.setAction(ACTION_UPDATE);
        if (statusText != null) i.putExtra("status_text", statusText);
        if (transcriptText != null) i.putExtra("transcript_text", transcriptText);
        try {
            ctx.startService(i);
        } catch (Exception ignored) {
        }
    }

    public static void hide(Context ctx) {
        Intent i = new Intent(ctx, PaiGenOverlayService.class);
        i.setAction(ACTION_HIDE);
        try {
            ctx.startService(i);
        } catch (Exception ignored) {
        }
    }

    private int dp(int v) {
        return (int) TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics()
        );
    }

    @Override
    public void onCreate() {
        super.onCreate();
        if (!Settings.canDrawOverlays(this)) {
            stopSelf();
            return;
        }
        isRunning = true;
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        buildOverlayViews();
        attachWindow();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (!Settings.canDrawOverlays(this)) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (intent != null) {
            String act = intent.getAction();
            if (ACTION_HIDE.equals(act)) {
                stopSelf();
                return START_NOT_STICKY;
            }
            if (intent.hasExtra("caller_name")) {
                currentCallerName = intent.getStringExtra("caller_name");
            }
            if (intent.hasExtra("caller_number")) {
                currentCallerNumber = intent.getStringExtra("caller_number");
            }
            if (intent.hasExtra("status_text")) {
                currentStatus = intent.getStringExtra("status_text");
            }
            if (intent.hasExtra("transcript_text")) {
                currentTranscript = intent.getStringExtra("transcript_text");
            }
            if (ACTION_SHOW.equals(act) && intent.hasExtra("expand")) {
                expanded = intent.getBooleanExtra("expand", false);
            }
            mainHandler.post(this::refreshUiState);
        }
        return START_STICKY;
    }

    private void buildOverlayViews() {
        rootContainer = new FrameLayout(this);

        // 1. Compact Floating ( ◠ ‿ ◠ ) Bubble
        bubbleView = new LinearLayout(this);
        bubbleView.setOrientation(LinearLayout.VERTICAL);
        bubbleView.setGravity(Gravity.CENTER);
        int bSize = dp(60);
        bubbleView.setLayoutParams(new FrameLayout.LayoutParams(bSize, bSize));
        GradientDrawable bBg = new GradientDrawable();
        bBg.setShape(GradientDrawable.OVAL);
        bBg.setColor(Color.parseColor("#F0000000"));
        bBg.setStroke(dp(2), Color.WHITE);
        bubbleView.setBackground(bBg);

        TextView faceTxt = new TextView(this);
        faceTxt.setText("(◠‿◠)");
        faceTxt.setTextColor(Color.WHITE);
        faceTxt.setTextSize(12);
        faceTxt.setTypeface(Typeface.DEFAULT_BOLD);
        faceTxt.setGravity(Gravity.CENTER);
        bubbleView.addView(faceTxt);

        bubbleSubLabel = new TextView(this);
        bubbleSubLabel.setText("AI");
        bubbleSubLabel.setTextColor(Color.parseColor("#BBBBBB"));
        bubbleSubLabel.setTextSize(9);
        bubbleSubLabel.setGravity(Gravity.CENTER);
        bubbleView.addView(bubbleSubLabel);

        // 2. Expanded Monochrome B&W Call + PC Floating HUD
        hudView = new LinearLayout(this);
        hudView.setOrientation(LinearLayout.VERTICAL);
        hudView.setPadding(dp(14), dp(12), dp(14), dp(12));
        hudView.setLayoutParams(new FrameLayout.LayoutParams(dp(296), FrameLayout.LayoutParams.WRAP_CONTENT));
        GradientDrawable hudBg = new GradientDrawable();
        hudBg.setColor(Color.parseColor("#F4000000"));
        hudBg.setStroke(dp(2), Color.WHITE);
        hudBg.setCornerRadius(dp(16));
        hudView.setBackground(hudBg);

        LinearLayout headerRow = new LinearLayout(this);
        headerRow.setOrientation(LinearLayout.HORIZONTAL);
        headerRow.setGravity(Gravity.CENTER_VERTICAL);

        txtOverlayStatus = new TextView(this);
        txtOverlayStatus.setLayoutParams(new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        txtOverlayStatus.setText("PAIGEN OVERLAY • READY");
        txtOverlayStatus.setTextColor(Color.WHITE);
        txtOverlayStatus.setTextSize(10);
        txtOverlayStatus.setTypeface(Typeface.DEFAULT_BOLD);
        headerRow.addView(txtOverlayStatus);

        Button btnMin = makeChipBtn("—", false);
        btnMin.setOnClickListener(v -> {
            expanded = false;
            refreshUiState();
        });
        headerRow.addView(btnMin);

        Button btnClose = makeChipBtn("✕", false);
        btnClose.setOnClickListener(v -> stopSelf());
        headerRow.addView(btnClose);
        hudView.addView(headerRow);

        txtOverlayCaller = new TextView(this);
        txtOverlayCaller.setTextColor(Color.WHITE);
        txtOverlayCaller.setTextSize(15);
        txtOverlayCaller.setTypeface(Typeface.DEFAULT_BOLD);
        txtOverlayCaller.setPadding(0, dp(6), 0, dp(2));
        hudView.addView(txtOverlayCaller);

        txtOverlayTranscript = new TextView(this);
        txtOverlayTranscript.setTextColor(Color.parseColor("#CCCCCC"));
        txtOverlayTranscript.setTextSize(12);
        txtOverlayTranscript.setPadding(dp(8), dp(6), dp(8), dp(6));
        GradientDrawable trBg = new GradientDrawable();
        trBg.setColor(Color.parseColor("#141414"));
        trBg.setStroke(dp(1), Color.parseColor("#2A2A2A"));
        trBg.setCornerRadius(dp(10));
        txtOverlayTranscript.setBackground(trBg);
        hudView.addView(txtOverlayTranscript);

        LinearLayout row1 = new LinearLayout(this);
        row1.setOrientation(LinearLayout.HORIZONTAL);
        row1.setPadding(0, dp(8), 0, 0);

        Button btnTalk = makeActionBtn("Let PaiGen Talk", true);
        btnTalk.setOnClickListener(v -> {
            Intent takeIntent = new Intent(this, MainActivity.class);
            takeIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            takeIntent.putExtra("action", "paigen_take_call");
            takeIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NAME, currentCallerName);
            takeIntent.putExtra(IncomingCallPromptActivity.EXTRA_CALLER_NUMBER, currentCallerNumber);
            startActivity(takeIntent);
        });
        row1.addView(btnTalk);

        Button btnEnd = makeActionBtn("End & Save", false);
        btnEnd.setOnClickListener(v -> {
            Intent endIntent = new Intent(this, MainActivity.class);
            endIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            endIntent.putExtra("action", "paigen_end_call");
            startActivity(endIntent);
            expanded = false;
            refreshUiState();
        });
        row1.addView(btnEnd);
        hudView.addView(row1);

        LinearLayout row2 = new LinearLayout(this);
        row2.setOrientation(LinearLayout.HORIZONTAL);
        row2.setPadding(0, dp(6), 0, 0);

        Button btnAsk = makeActionBtn("Voice Ask AI", false);
        btnAsk.setOnClickListener(v -> {
            Intent openIntent = new Intent(this, MainActivity.class);
            openIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            openIntent.putExtra("action", "paigen_voice_ask");
            startActivity(openIntent);
        });
        row2.addView(btnAsk);

        Button btnPc = makeActionBtn("PC Live Control", false);
        btnPc.setOnClickListener(v -> {
            Intent pcIntent = new Intent(this, MainActivity.class);
            pcIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            pcIntent.putExtra("action", "paigen_open_pc");
            startActivity(pcIntent);
        });
        row2.addView(btnPc);
        hudView.addView(row2);

        rootContainer.addView(bubbleView);
        rootContainer.addView(hudView);
        setupDragListener();
        refreshUiState();
    }

    private Button makeChipBtn(String label, boolean primary) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextSize(11);
        b.setAllCaps(false);
        b.setPadding(dp(8), dp(2), dp(8), dp(2));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(34), dp(28));
        lp.setMarginStart(dp(4));
        b.setLayoutParams(lp);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(primary ? Color.WHITE : Color.parseColor("#181818"));
        bg.setStroke(dp(1), Color.parseColor("#444444"));
        bg.setCornerRadius(dp(8));
        b.setBackground(bg);
        b.setTextColor(primary ? Color.BLACK : Color.WHITE);
        return b;
    }

    private Button makeActionBtn(String label, boolean primary) {
        Button b = new Button(this);
        b.setText(label);
        b.setTextSize(11);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setAllCaps(false);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, dp(36), 1f);
        lp.setMargins(dp(3), 0, dp(3), 0);
        b.setLayoutParams(lp);
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(primary ? Color.WHITE : Color.parseColor("#161616"));
        bg.setStroke(dp(1), primary ? Color.WHITE : Color.parseColor("#444444"));
        bg.setCornerRadius(dp(10));
        b.setBackground(bg);
        b.setTextColor(primary ? Color.BLACK : Color.WHITE);
        return b;
    }

    private void setupDragListener() {
        View.OnTouchListener dragTouch = new View.OnTouchListener() {
            private int initialX, initialY;
            private float initialTouchX, initialTouchY;
            private boolean moved;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        initialX = params.x;
                        initialY = params.y;
                        initialTouchX = event.getRawX();
                        initialTouchY = event.getRawY();
                        moved = false;
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        int dx = (int) (event.getRawX() - initialTouchX);
                        int dy = (int) (event.getRawY() - initialTouchY);
                        if (Math.abs(dx) > 8 || Math.abs(dy) > 8) moved = true;
                        params.x = initialX + dx;
                        params.y = initialY + dy;
                        if (windowManager != null && rootContainer != null) {
                            windowManager.updateViewLayout(rootContainer, params);
                        }
                        return true;
                    case MotionEvent.ACTION_UP:
                        if (!moved && v == bubbleView) {
                            expanded = true;
                            refreshUiState();
                        }
                        return true;
                }
                return false;
            }
        };
        bubbleView.setOnTouchListener(dragTouch);
        hudView.setOnTouchListener(dragTouch);
    }

    private void attachWindow() {
        int type = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
            ? WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            : WindowManager.LayoutParams.TYPE_PHONE;
        params = new WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
                | WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED,
            PixelFormat.TRANSLUCENT
        );
        params.gravity = Gravity.TOP | Gravity.START;
        params.x = dp(16);
        params.y = dp(140);
        windowManager.addView(rootContainer, params);
    }

    private void refreshUiState() {
        if (bubbleView == null || hudView == null) return;
        bubbleView.setVisibility(expanded ? View.GONE : View.VISIBLE);
        hudView.setVisibility(expanded ? View.VISIBLE : View.GONE);
        String callerDisp = (currentCallerNumber != null && !currentCallerNumber.isEmpty())
            ? currentCallerName + " (" + currentCallerNumber + ")"
            : currentCallerName;
        txtOverlayStatus.setText("PAIGEN OVERLAY • " + currentStatus);
        txtOverlayCaller.setText(callerDisp);
        txtOverlayTranscript.setText(currentTranscript);
        bubbleSubLabel.setText(currentStatus.contains("CALL") || currentStatus.contains("TALK") || currentStatus.contains("LISTEN") ? "CALL" : "AI");
    }

    @Override
    public void onDestroy() {
        isRunning = false;
        if (windowManager != null && rootContainer != null) {
            try {
                windowManager.removeView(rootContainer);
            } catch (Exception ignored) {
            }
        }
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
