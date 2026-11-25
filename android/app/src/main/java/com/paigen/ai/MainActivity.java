package com.paigen.ai;

import android.Manifest;
import android.app.Activity;
import android.app.role.RoleManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.media.AudioManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.provider.Settings;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.telecom.TelecomManager;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int REQ_PERMS = 101;
    private static final int REQ_ROLE_SCREENING = 102;

    private WebView webView;
    private CallVaultDbHelper dbHelper;
    private CallRecorderManager recorderManager;
    private SpeechRecognizer speechRecognizer;
    private TextToSpeech tts;
    private boolean ttsReady = false;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private boolean pageLoaded = false;
    private Intent pendingCallIntent = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        dbHelper = new CallVaultDbHelper(this);
        recorderManager = new CallRecorderManager(this);

        initTts();
        setupWebView();
        requestCorePermissions();
        startBridgeService();

        handleIncomingIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleIncomingIntent(intent);
    }

    private void handleIncomingIntent(Intent intent) {
        if (intent != null && "paigen_take_call".equals(intent.getStringExtra("action"))) {
            String name = intent.getStringExtra(IncomingCallPromptActivity.EXTRA_CALLER_NAME);
            String num = intent.getStringExtra(IncomingCallPromptActivity.EXTRA_CALLER_NUMBER);
            tryEngageHandsFreeCall();
            if (pageLoaded) {
                dispatchJsCallTakeover(name, num);
            } else {
                pendingCallIntent = intent;
            }
        }
    }

    private void tryEngageHandsFreeCall() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
                checkSelfPermission(Manifest.permission.ANSWER_PHONE_CALLS) == PackageManager.PERMISSION_GRANTED) {
                TelecomManager tm = (TelecomManager) getSystemService(Context.TELECOM_SERVICE);
                if (tm != null && tm.isInCall()) {
                    tm.acceptRingingCall();
                }
            }
        } catch (Exception ignored) {
        }
        try {
            AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
            if (am != null) {
                am.setMode(AudioManager.MODE_IN_COMMUNICATION);
                am.setSpeakerphoneOn(true);
            }
        } catch (Exception ignored) {
        }
    }

    private void dispatchJsCallTakeover(String name, String num) {
        String safeName = name == null ? "Incoming Caller" : name.replace("'", "");
        String safeNum = num == null ? "Unknown" : num.replace("'", "");
        mainHandler.post(() ->
            webView.evaluateJavascript(
                "window.onNativeCallTakeover && window.onNativeCallTakeover('" + safeName + "', '" + safeNum + "');",
                null
            )
        );
    }

    private void initTts() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                ttsReady = true;
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override
                    public void onStart(String utteranceId) {
                        notifySpeaking(true);
                    }
                    @Override
                    public void onDone(String utteranceId) {
                        notifySpeaking(false);
                    }
                    @Override
                    public void onError(String utteranceId) {
                        notifySpeaking(false);
                    }
                });
            }
        });
    }

    private void notifySpeaking(boolean speaking) {
        mainHandler.post(() -> {
            if (webView != null) {
                webView.evaluateJavascript("window.onNativeSpeakState && window.onNativeSpeakState(" + speaking + ");", null);
            }
        });
    }

    private void setupWebView() {
        webView = new WebView(this);
        WebSettings ws = webView.getSettings();
        ws.setJavaScriptEnabled(true);
        ws.setDomStorageEnabled(true);
        ws.setMediaPlaybackRequiresUserGesture(false);
        ws.setAllowFileAccess(true);
        ws.setAllowUniversalAccessFromFileURLs(true);
        ws.setAllowFileAccessFromFileURLs(true);
        ws.setMixedContentMode(WebSettings.MIXED_CONTENT_ALWAYS_ALLOW);
        webView.addJavascriptInterface(new PaiGenNativeBridge(), "PaiGenNative");
        webView.setWebChromeClient(new WebChromeClient());
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                pageLoaded = true;
                if (pendingCallIntent != null) {
                    handleIncomingIntent(pendingCallIntent);
                    pendingCallIntent = null;
                }
            }
        });
        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");
    }

    private void requestCorePermissions() {
        List<String> needed = new ArrayList<>();
        String[] perms = {
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.ANSWER_PHONE_CALLS,
            Manifest.permission.READ_CONTACTS,
            Manifest.permission.POST_NOTIFICATIONS
        };
        for (String p : perms) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU && Manifest.permission.POST_NOTIFICATIONS.equals(p)) {
                continue;
            }
            if (checkSelfPermission(p) != PackageManager.PERMISSION_GRANTED) {
                needed.add(p);
            }
        }
        if (!needed.isEmpty()) {
            requestPermissions(needed.toArray(new String[0]), REQ_PERMS);
        }
    }

    private void startBridgeService() {
        try {
            Intent svc = new Intent(this, PaiGenBridgeService.class);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(svc);
            } else {
                startService(svc);
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    protected void onDestroy() {
        if (speechRecognizer != null) speechRecognizer.destroy();
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        recorderManager.stopRecording();
        recorderManager.stopPlayback();
        super.onDestroy();
    }

    public class PaiGenNativeBridge {
        @JavascriptInterface
        public String getConfigJson() {
            SharedPreferences prefs = getSharedPreferences("paigen_prefs", MODE_PRIVATE);
            try {
                JSONObject o = new JSONObject();
                o.put("server_url", prefs.getString("server_url", "https://paigen-ai-cloud.onrender.com"));
                o.put("cloud_token", prefs.getString("cloud_token", ""));
                o.put("auto_ask_calls", prefs.getBoolean("auto_ask_calls", true));
                o.put("preferred_lang", prefs.getString("preferred_lang", "auto"));
                o.put("busy_mode", prefs.getBoolean("busy_mode", false));
                o.put("busy_timeout_sec", prefs.getInt("busy_timeout_sec", 8));
                o.put("forward_number", prefs.getString("forward_number", ""));
                return o.toString();
            } catch (Exception e) {
                return "{}";
            }
        }

        @JavascriptInterface
        public void saveConfig(String serverUrl, boolean autoAskCalls, String preferredLang) {
            saveConfigWithToken(serverUrl, "", autoAskCalls, preferredLang);
        }

        @JavascriptInterface
        public void saveConfigWithToken(String serverUrl, String cloudToken, boolean autoAskCalls, String preferredLang) {
            SharedPreferences prefs = getSharedPreferences("paigen_prefs", MODE_PRIVATE);
            prefs.edit()
                .putString("server_url", serverUrl)
                .putString("cloud_token", cloudToken == null ? "" : cloudToken.trim())
                .putBoolean("auto_ask_calls", autoAskCalls)
                .putString("preferred_lang", preferredLang)
                .apply();
        }

        @JavascriptInterface
        public void saveBusyConfig(boolean busyMode, int timeoutSec, String forwardNumber) {
            SharedPreferences prefs = getSharedPreferences("paigen_prefs", MODE_PRIVATE);
            prefs.edit()
                .putBoolean("busy_mode", busyMode)
                .putInt("busy_timeout_sec", timeoutSec)
                .putString("forward_number", forwardNumber == null ? "" : forwardNumber.trim())
                .apply();
        }

        @JavascriptInterface
        public void dialCallForwardCode(String ussdCode) {
            if (ussdCode == null || ussdCode.trim().isEmpty()) return;
            Intent i = new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + Uri.encode(ussdCode.trim())));
            startActivity(i);
        }

        @JavascriptInterface
        public void startSpeechListen(String langTag) {
            mainHandler.post(() -> {
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    requestCorePermissions();
                    return;
                }
                if (speechRecognizer != null) {
                    try { speechRecognizer.destroy(); } catch (Exception ignored) {}
                }
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(MainActivity.this);
                speechRecognizer.setRecognitionListener(new RecognitionListener() {
                    @Override public void onReadyForSpeech(Bundle params) {}
                    @Override public void onBeginningOfSpeech() {}
                    @Override public void onRmsChanged(float rmsdB) {}
                    @Override public void onBufferReceived(byte[] buffer) {}
                    @Override public void onEndOfSpeech() {}
                    @Override
                    public void onError(int error) {
                        webView.evaluateJavascript("window.onNativeSpeechError && window.onNativeSpeechError(" + error + ");", null);
                    }
                    @Override
                    public void onResults(Bundle results) {
                        ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                        String text = (matches != null && !matches.isEmpty()) ? matches.get(0) : "";
                        String safe = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ");
                        webView.evaluateJavascript("window.onNativeSpeechResult && window.onNativeSpeechResult('" + safe + "');", null);
                    }
                    @Override public void onPartialResults(Bundle partialResults) {}
                    @Override public void onEvent(int eventType, Bundle params) {}
                });
                Intent i = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                String locale = "en".equals(langTag) ? "en-IN" : "hi-IN";
                i.putExtra(RecognizerIntent.EXTRA_LANGUAGE, locale);
                i.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, locale);
                i.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1);
                speechRecognizer.startListening(i);
            });
        }

        @JavascriptInterface
        public void stopSpeechListen() {
            mainHandler.post(() -> {
                if (speechRecognizer != null) {
                    try { speechRecognizer.stopListening(); } catch (Exception ignored) {}
                }
            });
        }

        @JavascriptInterface
        public void speakNative(String text, String lang) {
            if (!ttsReady || text == null || text.isEmpty()) return;
            Locale loc = "hi".equals(lang) ? new Locale("hi", "IN") : Locale.UK;
            tts.setLanguage(loc);
            tts.setSpeechRate(1.02f);
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "paigen_" + System.currentTimeMillis());
        }

        @JavascriptInterface
        public void playAudioBase64(String b64) {
            try {
                notifySpeaking(true);
                recorderManager.playMp3Base64(b64, () -> notifySpeaking(false));
            } catch (Exception e) {
                notifySpeaking(false);
            }
        }

        @JavascriptInterface
        public String startCallRecording() {
            try {
                return recorderManager.startRecording();
            } catch (Exception e) {
                return "";
            }
        }

        @JavascriptInterface
        public String stopAndStoreCallRecording(
                String callerId,
                String callerName,
                String language,
                String triageTier,
                String psychState,
                String behavioralInsight,
                String transcriptJson,
                int synced
        ) {
            CallRecorderManager.RecordingResult res = recorderManager.stopRecording();
            long rowId = dbHelper.insertCallRecord(
                callerId, callerName, System.currentTimeMillis() / 1000L,
                res.durationSec, res.filePath, language, triageTier,
                psychState, behavioralInsight, transcriptJson, synced
            );
            try {
                JSONObject o = new JSONObject();
                o.put("id", rowId);
                o.put("audio_path", res.filePath);
                o.put("duration_sec", res.durationSec);
                o.put("audio_base64", CallRecorderManager.fileToBase64(res.filePath));
                return o.toString();
            } catch (Exception e) {
                return "{}";
            }
        }

        @JavascriptInterface
        public String getLocalCallVaultJson() {
            return dbHelper.listCallRecords(50).toString();
        }

        @JavascriptInterface
        public void playLocalRecording(String path) {
            try {
                notifySpeaking(true);
                recorderManager.playFile(path, () -> notifySpeaking(false));
            } catch (Exception e) {
                notifySpeaking(false);
            }
        }

        @JavascriptInterface
        public void stopAudioPlayback() {
            recorderManager.stopPlayback();
            notifySpeaking(false);
        }

        @JavascriptInterface
        public void simulateIncomingCall(String callerName, String callerNumber) {
            PaiGenBridgeService.triggerIncomingCallPrompt(MainActivity.this, callerName, callerNumber);
        }

        @JavascriptInterface
        public String getPermissionStatusJson() {
            try {
                JSONObject o = new JSONObject();
                o.put("mic", checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED);
                o.put("phone", checkSelfPermission(Manifest.permission.READ_PHONE_STATE) == PackageManager.PERMISSION_GRANTED);
                o.put("answer_calls", checkSelfPermission(Manifest.permission.ANSWER_PHONE_CALLS) == PackageManager.PERMISSION_GRANTED);
                o.put("overlay", Settings.canDrawOverlays(MainActivity.this));
                PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
                o.put("battery_unrestricted", pm != null && pm.isIgnoringBatteryOptimizations(getPackageName()));
                o.put("model", Build.MANUFACTURER + " " + Build.MODEL + " (Android " + Build.VERSION.RELEASE + ")");
                return o.toString();
            } catch (Exception e) {
                return "{}";
            }
        }

        @JavascriptInterface
        public void requestBatteryOptimizationExemption() {
            try {
                Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:" + getPackageName()));
                startActivity(i);
            } catch (Exception ignored) {
            }
        }

        @JavascriptInterface
        public void requestFullScreenIntentPermission() {
            if (Build.VERSION.SDK_INT >= 34) {
                try {
                    Intent i = new Intent("android.settings.MANAGE_APP_USE_FULL_SCREEN_INTENT", Uri.parse("package:" + getPackageName()));
                    startActivity(i);
                } catch (Exception e) {
                    openAppDetailsSettings();
                }
            } else {
                openAppDetailsSettings();
            }
        }

        @JavascriptInterface
        public void openAppDetailsSettings() {
            Intent i = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getPackageName()));
            startActivity(i);
        }

        @JavascriptInterface
        public void requestOverlayPermission() {
            if (!Settings.canDrawOverlays(MainActivity.this)) {
                Intent i = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + getPackageName()));
                startActivity(i);
            }
        }

        @JavascriptInterface
        public void requestCallScreeningRole() {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                RoleManager rm = (RoleManager) getSystemService(Context.ROLE_SERVICE);
                if (rm != null && rm.isRoleAvailable(RoleManager.ROLE_CALL_SCREENING) && !rm.isRoleHeld(RoleManager.ROLE_CALL_SCREENING)) {
                    startActivityForResult(rm.createRequestRoleIntent(RoleManager.ROLE_CALL_SCREENING), REQ_ROLE_SCREENING);
                }
            }
        }
    }
}
