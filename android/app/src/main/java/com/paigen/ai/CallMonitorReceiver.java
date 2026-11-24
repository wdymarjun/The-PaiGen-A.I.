package com.paigen.ai;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.telephony.TelephonyManager;

public class CallMonitorReceiver extends BroadcastReceiver {
    private static String lastHandledNumber = "";
    private static long lastHandledTs = 0L;

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null || !TelephonyManager.ACTION_PHONE_STATE_CHANGED.equals(intent.getAction())) {
            return;
        }

        SharedPreferences prefs = context.getSharedPreferences("paigen_prefs", Context.MODE_PRIVATE);
        boolean autoAsk = prefs.getBoolean("auto_ask_calls", true);
        if (!autoAsk) {
            return;
        }

        String state = intent.getStringExtra(TelephonyManager.EXTRA_STATE);
        if (TelephonyManager.EXTRA_STATE_RINGING.equals(state)) {
            String number = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);
            if (number == null || number.trim().isEmpty()) {
                number = "Private / Hidden Number";
            }

            long now = System.currentTimeMillis();
            if (number.equals(lastHandledNumber) && (now - lastHandledTs) < 8000L) {
                return;
            }
            lastHandledNumber = number;
            lastHandledTs = now;

            PaiGenBridgeService.triggerIncomingCallPrompt(context, "Incoming Caller", number);
        }
    }
}
