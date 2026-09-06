// 通知权限探测：UNUserNotificationCenter 当前负责进程的授权状态。
// 编译：swiftc -o <out> notify_probe.swift ；输出 JSON 行。
import Foundation
import UserNotifications

func statusName(_ s: UNAuthorizationStatus) -> String {
    switch s {
    case .notDetermined: return "notDetermined"
    case .denied: return "denied"
    case .authorized: return "authorized"
    case .provisional: return "provisional"
    case .ephemeral: return "ephemeral"
    @unknown default: return "unknown"
    }
}

func settingName(_ s: UNNotificationSetting) -> String {
    switch s {
    case .notSupported: return "notSupported"
    case .disabled: return "disabled"
    case .enabled: return "enabled"
    @unknown default: return "unknown"
    }
}

var done = false
let center = UNUserNotificationCenter.current()
center.getNotificationSettings { settings in
    let alert: String
    if #available(macOS 11.0, *) {
        alert = settingName(settings.alertSetting)
    } else {
        alert = settingName(settings.alertSetting)
    }
    let out: [String: Any] = [
        "authorizationStatus": statusName(settings.authorizationStatus),
        "alertSetting": alert,
        "badgeSetting": settingName(settings.badgeSetting),
        "soundSetting": settingName(settings.soundSetting),
        "lockScreenSetting": settingName(settings.lockScreenSetting),
        "notificationCenterSetting": settingName(settings.notificationCenterSetting),
    ]
    if let data = try? JSONSerialization.data(withJSONObject: out),
       let s = String(data: data, encoding: .utf8) {
        print(s)
    }
    done = true
}
// 完成回调可能依赖 run loop（XPC 回复）
let deadline = Date(timeIntervalSinceNow: 8)
while !done && RunLoop.current.run(mode: .default, before: deadline) {
    if Date() > deadline { break }
}
if !done {
    print(#"{"authorizationStatus":"timeout","alertSetting":"unknown"}"#)
}
exit(done ? 0 : 2)
