#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.

Sleep 4000
Iterations := 560

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

	Send %Iterations%
	Sleep 300

	Send {Enter}
	Sleep 500
	
	Sleep 3000
	Iterations+=1
}
esc::exitapp

