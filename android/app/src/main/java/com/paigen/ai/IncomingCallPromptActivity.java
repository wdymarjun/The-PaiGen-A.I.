package com.paigen.ai;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

public class IncomingCallPromptActivity extends Activity {
    public static final String EXTRA_CALLER_NAME = "caller_name";
    public static final String EXTRA_CALLER_NUMBER = "caller_number";

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
        if (callerName == null || callerName.isEmpty()) callerName = "Incoming Caller";
        if (callerNumber == null || callerNumber.isEmpty()) callerNumber = "Unknown Number";

        ((TextView) findViewById(R.id.txtCallerName)).setText(callerName);
        ((TextView) findViewById(R.id.txtCallerNumber)).setText(callerNumber);

        final String finalName = callerName;
        final String finalNum = callerNumber;

        Button btnTalk = findViewById(R.id.btnLetPaiGenTalk);
        Button btnSelf = findViewById(R.id.btnAnswerMyself);
        Button btnDismiss = findViewById(R.id.btnDismiss);

        btnTalk.setOnClickListener(v -> {
            Intent mainIntent = new Intent(this, MainActivity.class);
            mainIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            mainIntent.putExtra("action", "paigen_take_call");
            mainIntent.putExtra(EXTRA_CALLER_NAME, finalName);
            mainIntent.putExtra(EXTRA_CALLER_NUMBER, finalNum);
            startActivity(mainIntent);
            finish();
        });

        btnSelf.setOnClickListener(v -> finish());
        btnDismiss.setOnClickListener(v -> finish());
    }
}
