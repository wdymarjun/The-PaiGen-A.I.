package com.paigen.ai;

import android.content.SharedPreferences;
import android.telecom.Call;
import android.telecom.CallScreeningService;

public class PaiGenCallScreeningService extends CallScreeningService {
    @Override
    public void onScreenCall(Call.Details callDetails) {
        SharedPreferences prefs = getSharedPreferences("paigen_prefs", MODE_PRIVATE);
        boolean autoAsk = prefs.getBoolean("auto_ask_calls", true);

        String handle = callDetails.getHandle() != null
                ? callDetails.getHandle().getSchemeSpecificPart()
                : "Unknown Number";
        String callerName = callDetails.getCallerDisplayName();
        if (callerName == null || callerName.trim().isEmpty()) {
            callerName = "Incoming Caller";
        }

        if (autoAsk) {
            PaiGenBridgeService.triggerIncomingCallPrompt(this, callerName, handle);
        }

        CallResponse response = new CallResponse.Builder()
                .setDisallowCall(false)
                .setRejectCall(false)
                .setSilenceCall(false)
                .setSkipCallLog(false)
                .setSkipNotification(false)
                .build();
        respondToCall(callDetails, response);
    }
}
