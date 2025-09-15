import traceback
from datetime import datetime
from typing import List, Dict, Optional


class Logger:
    def __init__(self, max_logs: int = 1000):
        self.logs: List[Dict] = []
        self.max_logs = max_logs

    def _format_traceback(self, exception: Exception) -> str:
        try:
            if hasattr(exception, "__traceback__") and exception.__traceback__:
                return "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
            else:
                return f"{type(exception).__name__}: {str(exception)}"
        except:
            return f"{type(exception).__name__}: {str(exception)}"

    def _log(self, level: str, message: str, exception: Optional[Exception] = None):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        formatted_traceback = None
        if exception:
            formatted_traceback = self._format_traceback(exception)

        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "exception": str(exception) if exception else None,
            "traceback": formatted_traceback,
        }

        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

        print(f"[{timestamp}] {level}: {message}")
        if exception:
            print(f"Exception: {exception}")
            if formatted_traceback:
                print(f"Traceback:\n{formatted_traceback}")

    def info(self, message: str):
        self._log("INFO", message)

    def warning(self, message: str):
        self._log("WARNING", message)

    def error(self, message: str, exception: Optional[Exception] = None):
        self._log("ERROR", message, exception)

    def debug(self, message: str):
        self._log("DEBUG", message)

    def get_recent_logs(self, count: int = 50) -> List[Dict]:
        return self.logs[-count:] if self.logs else []
