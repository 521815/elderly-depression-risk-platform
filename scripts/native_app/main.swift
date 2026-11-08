import AppKit
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var backendProcess: Process?
    private var webappProcess: Process?
    private let backendHealth = URL(string: "http://127.0.0.1:8001/health")!
    private let webappHealth = URL(string: "http://localhost:8501/healthz")!
    private let webappURL = URL(string: "http://localhost:8501")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)

        let frame = NSRect(x: 0, y: 0, width: 1180, height: 780)
        window = NSWindow(
            contentRect: frame,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "老年抑郁风险筛查"
        window.center()

        webView = WKWebView(frame: frame)
        webView.navigationDelegate = self
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        startServicesAndLoad()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        backendProcess?.terminate()
        webappProcess?.terminate()
    }

    private func startServicesAndLoad() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            self.ensureBackend()
            self.ensureWebapp()

            for _ in 0..<120 {
                if self.isHealthy(self.backendHealth),
                   self.isHealthy(self.webappHealth) {
                    break
                }
                Thread.sleep(forTimeInterval: 0.5)
            }

            DispatchQueue.main.async { [weak self] in
                self?.loadWebApp()
            }
        }
    }

    private func ensureBackend() {
        guard !isHealthy(backendHealth) else { return }
        let project = URL(
            fileURLWithPath: "/Users/yellowich/Desktop/大创/老年抑郁风险预测大模型",
            isDirectory: true
        )
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/anaconda3/bin/python")
        process.arguments = [project.appendingPathComponent("backend_api/app.py").path]
        process.currentDirectoryURL = project
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            backendProcess = process
        } catch {
            print("Failed to start backend: \(error)")
        }
    }

    private func ensureWebapp() {
        guard !isHealthy(webappHealth) else { return }
        let project = URL(
            fileURLWithPath: "/Users/yellowich/Desktop/大创/老年抑郁风险预测大模型",
            isDirectory: true
        )
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/anaconda3/bin/streamlit")
        process.arguments = [
            "run",
            project.appendingPathComponent("webapp/app.py").path,
            "--server.headless",
            "true",
            "--server.port",
            "8501",
            "--server.fileWatcherType",
            "none",
        ]
        process.currentDirectoryURL = project
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            webappProcess = process
        } catch {
            print("Failed to start webapp: \(error)")
        }
    }

    private func isHealthy(_ url: URL) -> Bool {
        var request = URLRequest(url: url)
        request.timeoutInterval = 1.0
        let semaphore = DispatchSemaphore(value: 0)
        var healthy = false
        URLSession.shared.dataTask(with: request) { _, response, _ in
            healthy = (response as? HTTPURLResponse)?.statusCode == 200
            semaphore.signal()
        }.resume()
        _ = semaphore.wait(timeout: .now() + 2.0)
        return healthy
    }

    private func loadWebApp() {
        webView.load(URLRequest(url: webappURL))
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation?,
        withError error: Error
    ) {
        retryLoad()
    }

    func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation?,
        withError error: Error
    ) {
        retryLoad()
    }

    private func retryLoad() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.loadWebApp()
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
