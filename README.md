# Транскрибация видео в текст
![Logotype](https://timeweb.com/ru/community/article/a1/a179096fabed2b3a361f52471f1ed1a5.jpg)



# Иструкция по запуску приложения 

```
pip install -r requirements.txt
```
It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg
```
```
# on Arch Linux
sudo pacman -S ffmpeg
```
```
# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg
```
```
# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg
(Открывай командную строку с правами админа, иначе не установится и ничего работать не будет)
```
```
# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```
# Run the project
```
streamlit run main.py
```

