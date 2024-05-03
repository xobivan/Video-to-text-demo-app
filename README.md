# Транскрибация видео в текст
![Logotype](https://timeweb.com/ru/community/article/a1/a179096fabed2b3a361f52471f1ed1a5.jpg)

Прочитать текст по диагонали быстрее и эффективнее, чем переслушивать часовую запись конференции или интервью. Но подготовка протокола встречи тоже требует времени. Ускорить процесс позволит приложение, основаное на технологии машинного обучения, которое осуществяет перевод видео в текстовый формат. 

# Сферы применения транскрибации

### В обучении
В сфере онлайн-образования сервисы для расшифровки используют для создания текстовой версии видеоуроков. Некоторые ученики лучше воспринимают информацию визуально, чем на слух. Также программы для транскрибации нужны и в офлайн-образовании. Речь преподавателя на семинаре или конференции можно записать на диктофон, а потом расшифровать автоматически, чтобы не тратить много времени. Научным сотрудникам такие сервисы нужны, чтобы переводить в текст аудиоархивы — они становятся основой для монографий и статей.

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

