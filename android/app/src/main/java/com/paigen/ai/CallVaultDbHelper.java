package com.paigen.ai;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import org.json.JSONArray;
import org.json.JSONObject;

public class CallVaultDbHelper extends SQLiteOpenHelper {
    private static final String DB_NAME = "paigen_call_vault.db";
    private static final int DB_VERSION = 1;

    public CallVaultDbHelper(Context context) {
        super(context, DB_NAME, null, DB_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(
            "CREATE TABLE IF NOT EXISTS call_vault (" +
            "id INTEGER PRIMARY KEY AUTOINCREMENT," +
            "caller_id TEXT," +
            "caller_name TEXT," +
            "timestamp INTEGER," +
            "duration_sec REAL," +
            "audio_path TEXT," +
            "language TEXT," +
            "triage_tier TEXT," +
            "psych_state TEXT," +
            "behavioral_insight TEXT," +
            "transcript_json TEXT," +
            "synced INTEGER DEFAULT 0)"
        );
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS call_vault");
        onCreate(db);
    }

    public long insertCallRecord(
            String callerId,
            String callerName,
            long timestamp,
            double durationSec,
            String audioPath,
            String language,
            String triageTier,
            String psychState,
            String behavioralInsight,
            String transcriptJson,
            int synced
    ) {
        SQLiteDatabase db = getWritableDatabase();
        ContentValues cv = new ContentValues();
        cv.put("caller_id", callerId);
        cv.put("caller_name", callerName);
        cv.put("timestamp", timestamp);
        cv.put("duration_sec", durationSec);
        cv.put("audio_path", audioPath);
        cv.put("language", language);
        cv.put("triage_tier", triageTier);
        cv.put("psych_state", psychState);
        cv.put("behavioral_insight", behavioralInsight);
        cv.put("transcript_json", transcriptJson);
        cv.put("synced", synced);
        return db.insert("call_vault", null, cv);
    }

    public JSONArray listCallRecords(int limit) {
        JSONArray arr = new JSONArray();
        SQLiteDatabase db = getReadableDatabase();
        Cursor c = db.rawQuery(
            "SELECT * FROM call_vault ORDER BY timestamp DESC LIMIT ?",
            new String[]{String.valueOf(limit)}
        );
        try {
            while (c.moveToNext()) {
                JSONObject obj = new JSONObject();
                obj.put("id", c.getLong(c.getColumnIndexOrThrow("id")));
                obj.put("caller_id", c.getString(c.getColumnIndexOrThrow("caller_id")));
                obj.put("caller_name", c.getString(c.getColumnIndexOrThrow("caller_name")));
                obj.put("timestamp", c.getLong(c.getColumnIndexOrThrow("timestamp")));
                obj.put("duration_sec", c.getDouble(c.getColumnIndexOrThrow("duration_sec")));
                obj.put("audio_path", c.getString(c.getColumnIndexOrThrow("audio_path")));
                obj.put("language", c.getString(c.getColumnIndexOrThrow("language")));
                obj.put("triage_tier", c.getString(c.getColumnIndexOrThrow("triage_tier")));
                obj.put("psych_state", c.getString(c.getColumnIndexOrThrow("psych_state")));
                obj.put("behavioral_insight", c.getString(c.getColumnIndexOrThrow("behavioral_insight")));
                obj.put("transcript_json", c.getString(c.getColumnIndexOrThrow("transcript_json")));
                obj.put("synced", c.getInt(c.getColumnIndexOrThrow("synced")));
                arr.put(obj);
            }
        } catch (Exception ignored) {
        } finally {
            c.close();
        }
        return arr;
    }
}
