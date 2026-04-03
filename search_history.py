import os, time
import glob

history_path = os.path.expandvars(r"%APPDATA%\Code\User\History")
now = time.time()
cutoff = now - 6 * 3600  # 6 hours ago

found_files = []
try:
    for root, dirs, files in os.walk(history_path):
        for file in files:
            filepath = os.path.join(root, file)
            # Only check files modified in last 6 hours
            if os.path.getmtime(filepath) > cutoff:
                if file.endswith(".json"):
                    continue
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read(2000)
                        if "class MentorAgent" in content or "Agentic Data Science Tutor" in content or "pipeline_tracker" in content:
                            found_files.append(filepath)
                except Exception:
                    pass
    print("Found files:", found_files)
except Exception as e:
    print("Error:", e)
