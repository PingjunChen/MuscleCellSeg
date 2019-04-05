@echo on

if not exist "Log" do (
	mkdir Log
)
@echo ========> Log/copylog.txt

if not exist "build\bin\Debug" do (
	mkdir build\bin\Debug
)

::gtest
for /r ./DependLib/gtest/ %%i in ("/lib/*.dll") do (
	@echo %%~di%%~pilib\%%~ni%%~xi >> Log/copylog.txt
	copy "%%~di%%~pilib\%%~ni%%~xi" "./build/bin/Debug" >> Log/copylog.txt
)

::opencv
for /r ./DependLib/OpenCV-3.0.0 %%i in ("/bin/*.dll") do (
	@echo %%~di%%~pibin\%%~ni%%~xi >> Log/copylog.txt
	copy "%%~di%%~pibin\%%~ni%%~xi" "./build/bin/Debug" >> Log/copylog.txt
)
