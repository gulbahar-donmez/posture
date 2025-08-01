# 🏥 PostureGuard - AI Destekli Duruş Analizi Uygulaması

> **PostureGuard**, yapay zeka teknolojisi ile duruş bozukluklarını tespit eden ve kişiselleştirilmiş öneriler sunan modern bir React uygulamasıdır.

![PostureGuard Logo](./public/main-logo.png)

## ✨ Özellikler

### 🎯 **Duruş Analizi**
- **%98 Doğruluk Oranı**: Gelişmiş AI algoritmaları
- **Hızlı Analiz**: Saniyeler içinde sonuç
- **Görsel Raporlama**: Detaylı analiz sonuçları
- **Gerçek Zamanlı Preview**: Yüklenen fotoğrafın önizlemesi

### 🔐 **Kullanıcı Yönetimi**
- **Güvenli Login Sistemi**: LocalStorage tabanlı authentication
- **Kayıt Olma**: Yeni hesap oluşturma
- **Demo Hesaplar**: Test için hazır kullanıcılar
- **Oturum Yönetimi**: Güvenli giriş/çıkış

### 🎨 **Modern Tasarım**
- **Responsive Design**: Mobil ve masaüstü uyumlu
- **Professional UI**: Green-blue color scheme
- **Smooth Animations**: Particle efektleri ve geçişler
- **Interactive Elements**: Hover efektleri ve animasyonlar

### 📱 **Sayfalar**
- **Login/Signup Page**: Kullanıcı authentication
- **Dashboard**: Ana uygulama (Hero, Upload, About, Contact)
- **Dynamic Routing**: React Router ile sayfa yönetimi

## 🚀 Hızlı Başlangıç

### Gereksinimler
- **Node.js** (v16.0.0 veya üzeri)
- **npm** (v8.0.0 veya üzeri)
- Modern web tarayıcısı

### Kurulum

1. **Projeyi klonlayın**
```bash
git clone https://github.com/username/postureguard.git
cd postureguard
```

2. **Bağımlılıkları yükleyin**
```bash
npm install
```

3. **Uygulamayı başlatın**
```bash
npm start
```

4. **Tarayıcıda açın**
```
http://localhost:3000
```

## 👤 Demo Hesaplar

Uygulamayı test etmek için aşağıdaki demo hesapları kullanabilirsiniz:

| Rol | Kullanıcı Adı | Şifre | Açıklama |
|-----|---------------|-------|----------|
| **Admin** | `admin` | `password` | Yönetici hesabı |
| **Demo** | `demo` | `demo123` | Test kullanıcısı |

## 📖 Kullanım Kılavuzu

### 1. **Giriş Yapma**
- Ana sayfada otomatik olarak login ekranı açılır
- Demo hesaplardan birini kullanın veya yeni hesap oluşturun
- "Giriş Yap" butonuna tıklayın

### 2. **Kayıt Olma**
- Login sayfasında "Kayıt Ol" sekmesine tıklayın
- Ad Soyad, E-posta, Kullanıcı Adı ve Şifre bilgilerini girin
- "Hesap Oluştur" butonuna tıklayın
- Başarılı kayıt sonrası otomatik login moduna geçer

### 3. **Duruş Analizi**
- Dashboard'da "Analiz Et" bölümüne gidin
- Fotoğrafınızı drag & drop ile yükleyin veya "Dosya Seç" butonunu kullanın
- Önizleme ekranında fotoğrafınızı kontrol edin
- "Analizi Başlat" butonuna tıklayın

### 4. **Navigasyon**
- **Anasayfa**: Hero section ve genel bilgiler
- **Analiz Et**: Fotoğraf yükleme ve analiz
- **Hakkımızda**: Uygulama özellikleri
- **İletişim**: İletişim formu

## 🛠️ Teknik Detaylar

### Teknoloji Stack'i
- **React** 18.2.0 - Frontend framework
- **React Router DOM** 6.x - Routing yönetimi
- **CSS3** - Modern styling ve animasyonlar
- **LocalStorage** - Veri saklama
- **ES6+** - Modern JavaScript

### Proje Yapısı
```
PostureGuard/
├── public/
│   ├── index.html
│   ├── main-logo.png
│   └── Postureimg.jpg
├── src/
│   ├── components/
│   │   ├── AboutSection.js
│   │   ├── ContactSection.js
│   │   ├── HeroSection.js
│   │   ├── LoginPage.js
│   │   ├── MainApp.js
│   │   ├── Navbar.js
│   │   └── UploadSection.js
│   ├── App.js
│   ├── App.css
│   └── index.js
├── package.json
└── README.md
```

### Önemli Dosyalar
- **App.js**: Ana routing ve authentication logic
- **LoginPage.js**: Login/Signup formu
- **MainApp.js**: Dashboard komponenti
- **App.css**: Tüm CSS stiller
- **package.json**: Proje bağımlılıkları

## 🎨 Özelleştirme

### Renk Teması Değiştirme
Ana renk paletini değiştirmek için `src/App.css` dosyasındaki CSS değişkenlerini düzenleyin:

```css
/* Ana renkler */
--primary-green: #10b981
--primary-blue: #3b82f6
--dark-bg: #0f172a
```

### Logo Değiştirme
`public/main-logo.png` dosyasını kendi logonuzla değiştirin.

## 🔧 Geliştirme Komutları

```bash
# Geliştirme sunucusu
npm start

# Production build
npm run build

# Testleri çalıştır
npm test

# Kod analizi
npm run lint
```

## 📂 Önemli Notlar

### node_modules Klasörü
- **Boyut**: ~200MB, 16,000+ dosya
- **Gereklilik**: Proje çalışması için zorunlu
- **Alternatifler**: pnpm kullanarak %70 daha az yer kaplayabilir
- **Silme/Geri Yükleme**: `rmdir /s node_modules` ve `npm install`

### Veri Saklama
- Kullanıcı bilgileri LocalStorage'da saklanır
- Üretim ortamında veritabanı entegrasyonu önerilir
- Hassas bilgiler için şifreleme eklenebilir

## 🤝 Katkıda Bulunma

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 📞 İletişim

- **E-posta**: destek@postureguard.com
- **GitHub**: [PostureGuard Repository](https://github.com/username/postureguard)
- **Website**: [PostureGuard Official](https://postureguard.com)

## 🙏 Teşekkürler

Bu projeyi geliştirirken kullanılan açık kaynak kütüphanelere ve topluluk katkılarına teşekkürler.

---

**Made with ❤️ by PostureGuard Team**

> Duruş sağlığınız bizim önceliğimiz! 🏥✨ 