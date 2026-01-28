"""
FaceApp Windows Service
Runs FaceApp as a Windows service for automatic startup
"""

import os
import sys
import time
import subprocess
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket

class FaceAppService(win32serviceutil.ServiceFramework):
    _svc_name_ = "FaceAppService"
    _svc_display_name_ = "FaceApp Face Recognition Service"
    _svc_description_ = "Runs FaceApp face recognition application as a Windows service"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None
        self.log_file = None
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except:
                try:
                    self.process.kill()
                except:
                    pass

    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                              servicemanager.PYS_SERVICE_STARTED,
                              (self._svc_name_, ''))
        self.main()

    def main(self):
        # Change to the FaceApp directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)

        # Resolve venv python and prepare environment
        python_exe = os.path.join(app_dir, 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(python_exe):
            python_exe = sys.executable  # Fallback to current interpreter

        env = os.environ.copy()
        # Default to GPU processing for service unless explicitly overridden
        if 'PROCESSING_MODE' not in env or not env['PROCESSING_MODE']:
            env['PROCESSING_MODE'] = 'gpu'

        # Log service child process output to a file for diagnostics
        log_path = os.path.join(app_dir, 'service_output.log')
        try:
            self.log_file = open(log_path, 'ab')
        except Exception:
            self.log_file = None
        
        try:
            # Start the FaceApp process (production server)
            servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                                  servicemanager.PYS_SERVICE_STARTED,
                                  (self._svc_name_, f'Starting FaceApp using {python_exe} production.py'))
            
            # Launch server directly via venv python to avoid batch/shell issues in services
            creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            self.process = subprocess.Popen(
                [python_exe, 'production.py'],
                cwd=app_dir,
                stdout=self.log_file or subprocess.DEVNULL,
                stderr=self.log_file or subprocess.DEVNULL,
                env=env,
                creationflags=creation_flags,
            )
            
            # Wait for stop signal or process to end
            while True:
                # Check if service should stop
                if win32event.WaitForSingleObject(self.hWaitStop, 1000) == win32event.WAIT_OBJECT_0:
                    break
                
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process ended, log and restart
                    servicemanager.LogMsg(servicemanager.EVENTLOG_WARNING_TYPE,
                                          servicemanager.PYS_SERVICE_STARTED,
                                          (self._svc_name_, 'FaceApp process ended, restarting...'))
                    time.sleep(5)  # Wait before restart
                    
                    # Restart the process
                    self.process = subprocess.Popen(
                        [python_exe, 'production.py'],
                        cwd=app_dir,
                        stdout=self.log_file or subprocess.DEVNULL,
                        stderr=self.log_file or subprocess.DEVNULL,
                        env=env,
                        creationflags=creation_flags,
                    )
                    
        except Exception as e:
            servicemanager.LogMsg(servicemanager.EVENTLOG_ERROR_TYPE,
                                  servicemanager.PYS_SERVICE_STARTED,
                                  (self._svc_name_, f'Error: {str(e)}'))
        finally:
            try:
                if self.log_file:
                    self.log_file.flush()
            except Exception:
                pass

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(FaceAppService)