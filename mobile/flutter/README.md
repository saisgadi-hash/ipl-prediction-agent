# IPL Predictor — Flutter Mobile App

Cross-platform mobile app (iOS + Android) for the IPL Prediction Agent.

## Prerequisites

1. Install Flutter: https://docs.flutter.dev/get-started/install
2. Verify installation: `flutter doctor`

## Quick Start

```bash
cd mobile/flutter/ipl_predictor
flutter pub get
flutter run
```

## API Connection

The app connects to your Railway backend at `https://ipl-ai-prediction.up.railway.app`.
You can change this in the Settings screen within the app.

To run the API locally for development:
```bash
cd src/api
uvicorn main:app --reload --port 8000
```
Then update the API URL in Settings to `http://localhost:8000`.

## Build for Release

### Android APK
```bash
flutter build apk --release
```
Output: `build/app/outputs/flutter-apk/app-release.apk`

### iOS
```bash
flutter build ios --release
```
Requires Xcode and an Apple Developer account.

## Architecture

- `lib/main.dart` — App entry point and bottom navigation
- `lib/screens/` — All app screens (Home, Predict, Tournament, Settings)
- `lib/models/` — Data models for API responses
- `lib/services/` — API client service
- `lib/widgets/` — Reusable UI components (GlassCard)
- `lib/theme/` — Dark theme with IPL team colours
