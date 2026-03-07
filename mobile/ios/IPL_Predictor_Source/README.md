# iOS App Setup Instructions

The files in this folder (`ContentView.swift`, `PredictionViewModel.swift`, `PredictionModels.swift`, and `ResultCard.swift`) contain the complete SwiftUI code needed to run the iOS app on your Mac.

## How to run this App:

1. Open **Xcode** on your Mac.
2. Click **Create a new Xcode project**.
3. Select **iOS** -> **App** and click Next.
4. Fill in the details:
   - **Product Name:** `IPL Predictor`
   - **Interface:** `SwiftUI`
   - **Language:** `Swift`
5. Click Next and save the project somewhere on your computer.
6. Once the project opens, go to the left sidebar (Project Navigator).
7. Delete the default `ContentView.swift` file that Xcode created for you.
8. Drag and drop all 4 `.swift` files from this folder (`mobile/ios/IPL_Predictor_Source/`) into the left sidebar of your Xcode project to copy them in.
9. Make sure the FastAPI server is running in your terminal: `uvicorn src.api.main:app`
10. In Xcode, click the **Play** button at the top left to launch the iPhone Simulator and test the app!

*Note: The app expects the API to be running at `http://127.0.0.1:8000/predict_match`. If you test this on a real physical iPhone, you will need to change that URL inside `PredictionViewModel.swift` to match your Mac's local network IP address (e.g., `192.168.1.5`).*
