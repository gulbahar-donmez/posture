# 1. Adım: Temel Python imajını seçiyoruz.
FROM python:3.12-slim

# YENİ ADIM: OpenCV'nin ihtiyaç duyduğu TÜM sistem kütüphanelerini kuruyoruz.
# Bu, "libGL.so.1" ve "libgthread-2.0.so.0" hatalarını çözer.
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# 2. Adım: Çalışma dizinini belirliyoruz. Konteyner içinde kodlarımız burada olacak.
WORKDIR /app

# 3. Adım: Bağımlılıkları kurmak için önce sadece requirements.txt dosyasını kopyalıyoruz.
COPY requirements.txt .

# 4. Adım: Python bağımlılıklarını kuruyoruz.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Adım: Proje dosyalarının tamamını konteynerin içine kopyalıyoruz.
COPY . .

# 6. Adım: Uygulamayı çalıştıracak komutu belirtiyoruz.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
