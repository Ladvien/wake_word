#!/usr/bin/env python3
"""
Debug script to test Kokoro TTS API and diagnose silent audio files
"""
import requests
import json
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
import time
import subprocess
import sys


def test_direct_api():
    """Test the direct API call to Kokoro TTS"""
    print("🔍 Testing direct Kokoro TTS API...")

    # Test different API endpoints
    api_endpoints = [
        "https://hexgrad-kokoro-tts.hf.space/api/predict",
        "https://hexgrad-kokoro-tts.hf.space/call/predict",
        "https://hexgrad-kokoro-tts.hf.space/run/predict",
    ]

    test_text = "jade"
    test_voice = "af_sarah"
    test_speed = 1.0

    for api_url in api_endpoints:
        print(f"\n📡 Testing: {api_url}")

        try:
            # Try different payload formats
            payloads = [
                {"data": [test_text, test_voice, test_speed, 0, "wav"]},
                {"data": [test_text, test_voice, test_speed, 0]},
                {"inputs": test_text, "voice": test_voice, "speed": test_speed},
                {
                    "text": test_text,
                    "voice": test_voice,
                    "speed": test_speed,
                    "format": "wav",
                },
            ]

            for i, payload in enumerate(payloads):
                print(f"  🧪 Payload format {i+1}: {payload}")

                response = requests.post(api_url, json=payload, timeout=30)
                print(f"  📊 Status: {response.status_code}")
                print(f"  📊 Headers: {dict(response.headers)}")

                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(
                            f"  📊 Response keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
                        )
                        print(f"  📊 Response preview: {str(result)[:200]}...")

                        # Try to find audio data
                        if isinstance(result, dict):
                            if "data" in result:
                                print(f"  📊 Data field: {result['data']}")
                            if "url" in result:
                                print(f"  📊 URL field: {result['url']}")

                        return result

                    except json.JSONDecodeError:
                        print(
                            f"  ❌ Response not JSON. Content: {response.text[:200]}..."
                        )
                else:
                    print(f"  ❌ Error response: {response.text[:200]}...")

                time.sleep(1)  # Rate limit

        except Exception as e:
            print(f"  ❌ Exception: {e}")

    return None


def test_gradio_client():
    """Test using gradio_client library"""
    print("\n🔍 Testing with gradio_client...")

    try:
        from gradio_client import Client

        client = Client("https://hexgrad-kokoro-tts.hf.space/")
        print("✅ Connected to Gradio client")

        # Try to call the predict function
        result = client.predict(
            text="jade", voice="af_sarah", speed=1.0, pitch=0, api_name="/predict"
        )

        print(f"📊 Gradio result: {result}")
        return result

    except ImportError:
        print("❌ gradio_client not available. Install with: pip install gradio_client")
        return None
    except Exception as e:
        print(f"❌ Gradio client error: {e}")
        return None


def analyze_audio_file(file_path):
    """Analyze an audio file to check if it contains actual audio"""
    try:
        print(f"\n🔍 Analyzing audio file: {file_path}")

        # Check file size
        file_size = Path(file_path).stat().st_size
        print(f"📊 File size: {file_size} bytes")

        if file_size == 0:
            print("❌ File is empty!")
            return False

        # Try to load with librosa
        try:
            audio, sr = librosa.load(file_path, sr=None)
            print(f"📊 Sample rate: {sr} Hz")
            print(f"📊 Duration: {len(audio)/sr:.2f} seconds")
            print(f"📊 Audio shape: {audio.shape}")
            print(f"📊 Audio range: [{np.min(audio):.6f}, {np.max(audio):.6f}]")
            print(f"📊 RMS energy: {np.sqrt(np.mean(audio**2)):.6f}")

            # Check if audio is silent
            rms_threshold = 0.001
            is_silent = np.sqrt(np.mean(audio**2)) < rms_threshold
            print(f"📊 Is silent (RMS < {rms_threshold}): {is_silent}")

            return not is_silent

        except Exception as e:
            print(f"❌ librosa load error: {e}")

        # Try with soundfile
        try:
            audio, sr = sf.read(file_path)
            print(f"📊 soundfile - Sample rate: {sr} Hz, shape: {audio.shape}")
            return True

        except Exception as e:
            print(f"❌ soundfile error: {e}")
            return False

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def test_file_download(url):
    """Test downloading a file from URL"""
    print(f"\n🔍 Testing file download from: {url}")

    try:
        response = requests.get(url, timeout=30)
        print(f"📊 Download status: {response.status_code}")
        print(f"📊 Content-Type: {response.headers.get('content-type')}")
        print(f"📊 Content-Length: {response.headers.get('content-length')}")

        if response.status_code == 200:
            # Save to temp file
            temp_file = Path("temp_kokoro_test.wav")
            with open(temp_file, "wb") as f:
                f.write(response.content)

            print(f"📊 Downloaded {len(response.content)} bytes")

            # Analyze the downloaded file
            has_audio = analyze_audio_file(temp_file)

            # Clean up
            if temp_file.exists():
                temp_file.unlink()

            return has_audio
        else:
            print(f"❌ Download failed: {response.text[:200]}...")
            return False

    except Exception as e:
        print(f"❌ Download error: {e}")
        return False


def check_existing_files():
    """Check existing generated files"""
    print("\n🔍 Checking existing generated files...")

    # Look for files in the output directory
    output_dirs = [
        Path("./output/data/raw_positive"),
        Path("./output/data/training/positive"),
        Path("./positive_samples"),
        Path("."),
    ]

    for output_dir in output_dirs:
        if output_dir.exists():
            wav_files = list(output_dir.glob("*.wav"))
            print(f"📁 {output_dir}: {len(wav_files)} WAV files")

            for wav_file in wav_files[:3]:  # Check first 3 files
                has_audio = analyze_audio_file(wav_file)
                print(f"  {'✅' if has_audio else '❌'} {wav_file.name}")


def test_alternative_tts():
    """Test alternative TTS as backup"""
    print("\n🔍 Testing alternative TTS options...")

    # Test system TTS (if available)
    try:
        import pyttsx3

        print("✅ pyttsx3 available")

        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        print(f"📊 Available voices: {len(voices) if voices else 0}")

        if voices:
            # Test generating a sample
            test_file = Path("test_pyttsx3.wav")
            engine.save_to_file("jade", str(test_file))
            engine.runAndWait()

            if test_file.exists():
                has_audio = analyze_audio_file(test_file)
                test_file.unlink()
                print(f"📊 pyttsx3 test: {'✅ Success' if has_audio else '❌ Silent'}")
            else:
                print("❌ pyttsx3 failed to create file")

    except ImportError:
        print("❌ pyttsx3 not available")
    except Exception as e:
        print(f"❌ pyttsx3 error: {e}")


def main():
    """Main debug function"""
    print("🐛 Kokoro TTS Debug Script")
    print("=" * 50)

    # Check internet connectivity
    try:
        response = requests.get("https://httpbin.org/ip", timeout=5)
        print(f"✅ Internet connection: {response.json()}")
    except Exception as e:
        print(f"❌ Internet connection issue: {e}")
        return

    # Test 1: Check existing files
    check_existing_files()

    # Test 2: Direct API test
    api_result = test_direct_api()

    # Test 3: Gradio client test
    gradio_result = test_gradio_client()

    # Test 4: Alternative TTS
    test_alternative_tts()

    # Summary
    print("\n" + "=" * 50)
    print("🔍 DEBUG SUMMARY:")
    print(f"  • API test: {'✅' if api_result else '❌'}")
    print(f"  • Gradio test: {'✅' if gradio_result else '❌'}")

    if api_result:
        print(f"  • API result preview: {str(api_result)[:100]}...")

    print("\n💡 RECOMMENDATIONS:")
    if not api_result and not gradio_result:
        print("  1. Try installing gradio_client: pip install gradio_client")
        print("  2. Check if the Hugging Face Space is working in browser")
        print("  3. Consider using the direct Kokoro library instead of API")
        print("  4. Use alternative TTS as fallback")
    else:
        print("  1. API is working, check file download and processing")
        print("  2. Verify audio file format and content")


if __name__ == "__main__":
    main()
