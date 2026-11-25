package com.paigen.ai;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.telephony.TelephonyManager;

public class CallMonitorReceiver extends BroadcastReceiver {
    private static volatile String lastNumber = "Private / Hidden Number";
    private static volatile long lastRingTs = 0L;
    private static volatile boolean isRinging = false;
    private static volatile boolean wasOffhook = false;

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null || !TelephonyManager.ACTION_PHONE_STATE_CHANGED.equals(intent.getAction())) {
            return;
        }

        SharedPreferences prefs = context.getSharedPreferences("paigen_prefs", Context.MODE_PRIVATE);
        boolean autoAsk = prefs.getBoolean("auto_ask_calls", true);
        boolean busyMode = prefs.getBoolean("busy_mode", false);
        if (!autoAsk && !busyMode) {
            return;
        }

        String state = intent.getStringExtra(TelephonyManager.EXTRA_STATE);
        if (TelephonyManager.EXTRA_STATE_RINGING.equals(state)) {
            String number = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER);
            if (number != null && !number.trim().isEmpty()) {
                lastNumber = number.trim();
            }
            long now = System.currentTimeMillis();
            if (isRinging && (now - lastRingTs) < 5000L) {
                return;
            }
            isRinging = true;
            wasOffhook = false;
            lastRingTs = now;

            PaiGenBridgeService.onIncomingRing(context, "Incoming Caller", lastNumber);
        } else if (TelephonyManager.EXTRA_STATE_OFFHOOK.equals(state)) {
            wasOffhook = true;
            isRinging = false;
            PaiGenBridgeService.cancelAutoBusyTimer();
        } else if (TelephonyManager.EXTRA_STATE_IDLE.equals(state)) {
            if (isRinging && !wasOffhook) {
                // Caller hung up unanswered OR Arjun pressed Decline -> trigger Busy Mode handoff
                PaiGenBridgeService.onCallDeclinedOrMissed(context, lastNumber);
            }
            isRinging = false;
            wasOffhook = false;
        }
    }
}
