/**
 * OfflineBridgeService — Android background service for tethering to Bubby.
 *
 * Establishes a persistent TCP connection to the local bridge server running
 * on the companion Linux desktop. Transmits device state (battery, active app,
 * notifications) as JSON-line payloads. Operates completely offline — no cloud
 * APIs, no external internet. Connection via local Wi-Fi hotspot or USB tether.
 *
 * Architecture:
 * 1. Service starts on boot or manual launch
 * 2. Opens TCP socket to desktop IP (configurable)
 * 3. Registers BroadcastReceivers for battery, app changes
 * 4. Sends JSON payloads every 30s (battery) + on app change events
 * 5. Reads acknowledgments from the server
 *
 * Build: Compile with Android SDK 24+ (no external libraries needed)
 *        Uses only standard Android SDK + kotlinx.coroutines
 *
 * Permissions needed in AndroidManifest.xml:
 *   <uses-permission android:name="android.permission.INTERNET" />
 *   <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
 *   <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
 *   <uses-permission android:name="android.permission.QUERY_ALL_PACKAGES" />
 *
 * To run as a standalone APK, wrap this service in a minimal Activity
 * that starts it and provides server IP configuration.
 */

package com.bubby.bridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.Socket

class OfflineBridgeService : Service() {

    companion object {
        const val TAG = "BubbyBridge"
        const val CHANNEL_ID = "bubby_bridge_channel"
        const val NOTIFICATION_ID = 1001
        
        // Configuration — change these for your network setup
        const val DEFAULT_SERVER_HOST = "192.168.1.100"  // Desktop IP
        const val SERVER_PORT = 9877
        
        // Polling intervals (milliseconds)
        const val BATTERY_POLL_INTERVAL = 30_000L   // 30 seconds
        const val RECONNECT_DELAY = 10_000L          // 10 seconds
    }

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var serverHost: String = DEFAULT_SERVER_HOST
    private var socket: Socket? = null
    private var writer: OutputStreamWriter? = null
    private var reader: BufferedReader? = null
    private var isConnected: Boolean = false

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Bridge service created")
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Connecting..."))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // Allow user to override server IP via intent extras
        intent?.getStringExtra("server_host")?.let { host ->
            serverHost = host
            Log.i(TAG, "Server host set to: $serverHost")
        }

        // Start the connection loop
        serviceScope.launch {
            connectAndStream()
        }

        // Register battery receiver
        registerBatteryReceiver()

        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        Log.i(TAG, "Bridge service destroyed")
        serviceScope.cancel()
        disconnect()
        super.onDestroy()
    }

    /**
     * Main connection loop — attempts to connect, then streams data.
     * Auto-reconnects on disconnection.
     */
    private suspend fun connectAndStream() {
        while (isActive) {
            try {
                Log.i(TAG, "Connecting to $serverHost:$SERVER_PORT...")
                socket = Socket(serverHost, SERVER_PORT)
                writer = OutputStreamWriter(socket!!.getOutputStream())
                reader = BufferedReader(InputStreamReader(socket!!.getInputStream()))
                isConnected = true
                
                Log.i(TAG, "Connected to bridge server")
                updateNotification("Connected to Bubby")

                // Send initial device info
                sendEvent(buildDeviceInfo())

                // Start periodic battery reporting
                launch { batteryReportingLoop() }

                // Listen for server acknowledgments
                listenForAcks()

            } catch (e: Exception) {
                Log.e(TAG, "Connection failed: ${e.message}")
                isConnected = false
                updateNotification("Disconnected — retrying...")
                delay(RECONNECT_DELAY)
            }
        }
    }

    /**
     * Periodically send battery status updates.
     */
    private suspend fun batteryReportingLoop() {
        while (isConnected && isActive) {
            try {
                val batteryData = getBatteryStatus()
                sendEvent(batteryData)
                Log.d(TAG, "Battery update sent: ${batteryData.optInt("device_battery")}%")
            } catch (e: Exception) {
                Log.e(TAG, "Battery report failed: ${e.message}")
                break
            }
            delay(BATTERY_POLL_INTERVAL)
        }
    }

    /**
     * Listen for JSON acknowledgment lines from the server.
     */
    private suspend fun listenForAcks() {
        try {
            while (isConnected && isActive) {
                val line = withContext(Dispatchers.IO) {
                    reader?.readLine()
                } ?: break

                if (line.isNotBlank()) {
                    Log.d(TAG, "Server ACK: $line")
                    try {
                        val response = JSONObject(line)
                        val status = response.optString("status", "unknown")
                        if (status == "error") {
                            Log.w(TAG, "Server reported error: ${response.optString("reason")}")
                        }
                    } catch (e: Exception) {
                        // Non-JSON response — just log it
                        Log.d(TAG, "Server: $line")
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "ACK listener error: ${e.message}")
        } finally {
            isConnected = false
            Log.i(TAG, "Disconnected from server")
            updateNotification("Disconnected")
        }
    }

    /**
     * Send a JSON event to the server.
     */
    private suspend fun sendEvent(data: JSONObject) {
        if (!isConnected) return
        
        try {
            withContext(Dispatchers.IO) {
                writer?.apply {
                    write(data.toString() + "\n")
                    flush()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to send event: ${e.message}")
            isConnected = false
        }
    }

    /**
     * Build the initial device info payload.
     */
    private fun buildDeviceInfo(): JSONObject {
        return JSONObject().apply {
            put("type", "device_info")
            put("device_id", "${Build.MANUFACTURER}_${Build.MODEL}".lowercase().replace(" ", "_"))
            put("device_name", "${Build.MANUFACTURER} ${Build.MODEL}")
            put("android_version", Build.VERSION.RELEASE)
            put("sdk_version", Build.VERSION.SDK_INT)
            put("timestamp", System.currentTimeMillis() / 1000.0)
        }
    }

    /**
     * Get current battery status from the system.
     */
    private fun getBatteryStatus(): JSONObject {
        val batteryIntent = registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
        
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: 100
        val batteryPct = if (scale > 0) (level * 100 / scale) else 0
        
        val status = batteryIntent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING
                || status == BatteryManager.BATTERY_STATUS_FULL

        return JSONObject().apply {
            put("type", "battery_update")
            put("device_id", "${Build.MANUFACTURER}_${Build.MODEL}".lowercase())
            put("device_battery", batteryPct)
            put("device_charging", isCharging)
            put("timestamp", System.currentTimeMillis() / 1000.0)
            put("message", "Battery at $batteryPct%${if (isCharging) " (charging)" else ""}")
        }
    }

    /**
     * Register a broadcast receiver for battery changes.
     */
    private fun registerBatteryReceiver() {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                if (intent?.action == Intent.ACTION_BATTERY_CHANGED) {
                    if (isConnected) {
                        serviceScope.launch {
                            try {
                                sendEvent(getBatteryStatus())
                            } catch (e: Exception) {
                                Log.e(TAG, "Battery broadcast send failed: ${e.message}")
                            }
                        }
                    }
                }
            }
        }
        
        registerReceiver(
            receiver,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
    }

    /**
     * Disconnect from the server.
     */
    private fun disconnect() {
        isConnected = false
        try { writer?.close() } catch (_: Exception) {}
        try { reader?.close() } catch (_: Exception) {}
        try { socket?.close() } catch (_: Exception) {}
        socket = null
        writer = null
        reader = null
    }

    /**
     * Create the notification channel (required for foreground service).
     */
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Bubby Bridge",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Connection status for Bubby desktop companion"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    /**
     * Build a foreground service notification.
     */
    private fun buildNotification(text: String): Notification {
        // In a real app, this would open a configuration Activity
        val pendingIntent = PendingIntent.getActivity(
            this, 0, Intent(), PendingIntent.FLAG_IMMUTABLE
        )
        
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("Bubby Bridge")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_share)
                .setContentIntent(pendingIntent)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("Bubby Bridge")
                .setContentText(text)
                .setSmallIcon(android.R.drawable.ic_menu_share)
                .setContentIntent(pendingIntent)
                .build()
        }
    }

    /**
     * Update the foreground notification text.
     */
    private fun updateNotification(text: String) {
        val notification = buildNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }
}

/*
 * ============================================================================
 * ANDROID SETUP INSTRUCTIONS
 * ============================================================================
 *
 * 1. Add service declaration to AndroidManifest.xml:
 *    <service
 *        android:name=".OfflineBridgeService"
 *        android:exported="false"
 *        android:foregroundServiceType="dataSync" />
 *
 * 2. Request permissions in AndroidManifest.xml:
 *    <uses-permission android:name="android.permission.INTERNET" />
 *    <uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
 *    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
 *
 * 3. Start the service from an Activity:
 *    val intent = Intent(this, OfflineBridgeService::class.java).apply {
 *        putExtra("server_host", "192.168.1.100")  // Desktop IP
 *    }
 *    startForegroundService(intent)
 *
 * 4. To connect via USB tethering (no Wi-Fi needed):
 *    a. Enable USB tethering on phone
 *    b. Connect USB cable to laptop
 *    c. Server IP is typically 192.168.42.x (check with `ip addr`)
 *
 * 5. To connect via local Wi-Fi hotspot:
 *    a. Create hotspot on laptop: `nmcli dev wifi hotspot ifname wlan0 ssid Bubby password 12345678`
 *    b. Connect phone to "Bubby" Wi-Fi
 *    c. Server IP is the hotspot gateway (typically 10.42.0.1 or 192.168.12.1)
 */