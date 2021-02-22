import subprocess
import sys
import PySimpleGUI as sg

def main():
    layout = [  [sg.Text('Enter a command to execute (e.g. dir or ls)')],
                [sg.Input(key='_IN_')],
                [sg.Output(size=(60,15))],
                [sg.Button('Run'), sg.Button('Exit')] ]

    window = sg.Window('Realtime Shell Command Output', layout)

    while True:             # Event Loop
        event, values = window.Read()
        if event in (None, 'Exit'):
            break

        if event == 'Run':
            runCommand(cmd=values['_IN_'], window=window)

    window.Close()


def runCommand(cmd, timeout=None, window=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''
    for line in p.stdout:
        line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
        output += line
        print(line)
        window.Refresh() if window else None
    retval = p.wait(timeout)
    return (retval, output)

if __name__ == '__main__':
    main()
