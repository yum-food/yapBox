param(
  [switch]$skip_zip = $false,
  [string]$release = "Release",
  [string]$install_pip = $true
)

echo "Skip zip: $skip_zip"
echo "Release: $release"
echo "Install pip: $install_pip"

$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$install_dir = "yapBox"

if (Test-Path $install_dir) {
  rm -Recurse -Force $install_dir
}

$py_dir = "Python"

if (Test-Path $py_dir) {
  rm -Recurse $py_dir
}
if (-Not (Test-Path $py_dir)) {
  echo "Fetching python"

  $PYTHON_3_10_9_URL = "https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip"
  $PYTHON_URL = $PYTHON_3_10_9_URL
  $PYTHON_FILE = $(Split-Path -Path $PYTHON_URL -Leaf)

  if (-Not (Test-Path $PYTHON_FILE)) {
    Invoke-WebRequest $PYTHON_URL -OutFile $PYTHON_FILE
  }

  mkdir Python
  Expand-Archive $PYTHON_FILE -DestinationPath Python

  echo ".." >> Python/python310._pth
  echo "import site" >> Python/python310._pth
}

$pip_path = "$py_dir/get-pip.py"

if (Test-Path $pip_path) {
  rm -Force $pip_path
}

if (-Not (Test-Path $pip_path)) {
  echo "Fetching pip"

  $PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
  $PIP_FILE = $(Split-Path -Path $PIP_URL -Leaf)

  if (-Not (Test-Path $PIP_FILE)) {
    Invoke-WebRequest $PIP_URL -OutFile $PIP_FILE
  }

  mv $PIP_FILE $pip_path
}

if ($install_pip) {
  ./Python/python.exe Python/get-pip.py

  echo "Installing requirements"
  echo "Assuming host has python 3.10.9 installed" # TODO test for this
  python -m pip install -r ../requirements.txt --target Python/Lib/site-packages
}

if (-Not (Test-Path "silero-vad")) {
  git clone "https://github.com/snakers4/silero-vad"
}

mkdir $install_dir > $null
mkdir $install_dir/Models
cp ../*.py $install_dir/
cp ../*.bat $install_dir/
cp ../*.txt $install_dir/
cp -Recurse Python $install_dir/Python
cp "silero-vad/files/silero_vad.onnx" $install_dir/Models/
cp "silero-vad/LICENSE" $install_dir/Models/silero_vad.onnx.LICENSE

if (-Not $skip_zip) {
  Compress-Archive -Path "$install_dir" -DestinationPath "$install_dir.zip" -Force
}

