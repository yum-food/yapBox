## yapBox

A black box for your yapping.

This app records your mic. It uses silero-vad to split audio into contiguous
segments of speech, and saves them to disk as .wav files. Metadata is
saved to a corresponding .yaml file.

What's a black box? Wikipedia says this:
```
A flight recorder is an electronic recording device placed in an aircraft for
the purpose of facilitating the investigation of aviation accidents and
incidents. The device may often be referred to colloquially as a "black box",
an outdated name which has become a misnomer—they are now required to be
painted bright orange, to aid in their recovery after accidents.
```

This is a CLI app. It is not polished and requires a little elbow grease to
use properly. The intent is to assist people who want to gather high-quality
training data of human voices. Use responsibly.

## Compatibility

This application is designed for Windows 10. Functionality on any other
platform is purely coincidental.

## Running

Download the latest release and double click `app.bat` in File Explorer.

Read the output and change the mic to whatever you're using. To change mics,
edit app.py. Any text editor works, including Notepad.

## Building from source

First install python 3.10.9. Make sure that Powershell is using that version by
typing this (leave out the $, it's used to differentiate between commands and
output):
```
$ python.exe --version
Python 3.10.9
```

Then open Powershell and run package.ps1:
```
$ cd pkg
$ ./package.ps1
```

All dependencies should download themselves. It will use the host Python to
install dependencies into the app's environment.

## Ethics

We are living in the wild west of AI. You can clone anyone's voice and
plausibly reproduce it using projects like
[rvc-beta](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases).
Legislation has not caught
up to this yet. Cloning someone's voice without their consent is, at best,
ethically dubious. This tool makes that process easier. In the absence of a
legal framework, you must make your own choices as to what is right. Take this
seriously. When in doubt, follow Kant's [universalization
principle](https://en.wikipedia.org/wiki/Universalizability) and the [golden
rule](https://en.wikipedia.org/wiki/Golden_Rule).

