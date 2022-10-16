#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

Sleep 5000

Loop
{
	Send {LAlt}
	Sleep 300
	Send f
	Sleep 300
	Send s
	Sleep 300
	Send s
	Sleep 300

	Send a
	Sleep 300

	Send {Enter}
	Sleep 500
	Send {Left}
	Sleep 300
	Send {Enter}
	Sleep 7000
}
esc::exitapp

