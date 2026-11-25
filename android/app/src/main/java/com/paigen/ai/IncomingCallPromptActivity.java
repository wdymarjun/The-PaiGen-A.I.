package com.paigen.ai;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

public class IncomingCallPromptActivity extends Activity {
    public static final String EXTRA_CALLER_NAME = "caller_name";
    public static final String EXTRA_CALLER_NUMBER = "caller_number";

    private final Handler handler = new Handler(Looper.getMainLooper());
    private Runnable countdownTick;
    private int secondsRemaining = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
        } else {
            getWindow().addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON |
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
            );
        }

        setContentView(R.layout.activity_incoming_prompt);

        String callerName = getIntent().getStringExtra(EXTRA_CALLER_NAME);
        String callerNumber = getIntent().getStringExtra(EXTRA_CALLER_NUMBER);
        boolean busyMode = getIntent().getBooleanExtra("busy_mode", false);
        secondsRemaining = getIntent().getIntExtra("busy_timeout_sec", 8);

        if (callerName == null || callerName.isEmpty()) callerName = "Incoming Caller";
        if (callerNumber == null || callerNumber.isEmpty()) callerNumber = "Unknown Number";

        ((TextView) findViewById(R.id.txtCallerName)).setText(callerName);
        ((TextView) findViewById(R.id.txtCallerNumber)).setText(callerNumber);

        final TextView txtBadge = findViewById(R.id.txtBadge);
        final Button btnTalk = findViewById(R.id.btnLetPaiGenTalk);
        final Button btnSelf = findViewById(R.id.btnAnswerMyself);
        final Button btnDismiss = findViewById(R.id.btnDismiss);

        final String finalName = callerName;
        final String finalNum = callerNumber;

        Runnable launchTakeover = () -> {
            stopCountdown();
            PaiGenBridgeService.cancelAutoBusyTimer();
            Intent mainIntent = new Intent(this, MainActivity.class);
            mainIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            mainIntent.putExtra("action", "paigen_take_call");
            mainIntent.putExtra(EXTRA_CALLER_NAME, finalName);
            mainIntent.putExtra(EXTRA_CALLER_NUMBER, finalNum);
            startActivity(mainIntent);
            finish();
        };

        btnTalk.setOnClickListener(v -> launchTakeover.run());
        btnSelf.setOnClickListener(v -> {
            stopCountdown();
            PaiGenBridgeService.cancelAutoBusyTimer();
            finish();
        });
        btnDismiss.setOnClickListener(v -> {
            stopCountdown();
            PaiGenBridgeService.cancelAutoBusyTimer();
            finish();
        });

        if (busyMode) {
            if (secondsRemaining <= 0) {
                launchTakeover.run();
                return;
            }
            countdownTick = new Runnable() {
                @Override
                public void run() {
                    if (secondsRemaining <= 0) {
                        launchTakeover.run();
                        return;
                    }
                    txtBadge.setText("100% AUTO-BUSY ACTIVE • TAKING CALL IN " + secondsRemaining + "s");
                    btnTalk.setText("Let PaiGen Talk Now (" + secondsRemaining + "s)");
                    secondsRemaining--;
                    handler.postDelayed(this, 1000L);
                }
            };
            handler.post(countdownTick);
        }
    }

    private void stopCountdown() {
        if (countdownTick != null) {
            handler.removeCallbacks(countdownTick);
            countdownTick = null;
        }
    }

    @Override
    protected void onDestroy() {
        stopCountdown();
        super.onDestroy();
    }
}
