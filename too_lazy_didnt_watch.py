import os
import torch
from pytube import YouTube
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import anthropic
import json
import re

def format_timestamp(timestamp):
    if timestamp is None:
        return "unknown"
    return f"{timestamp:.2f}"

def create_output_directory(video_id):
    dir_name = f"output_{video_id}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def download_video(url, output_dir):
    yt = YouTube(url)
    print("Downloading video...")
    video_stream = yt.streams.get_highest_resolution()
    video_file = video_stream.download(output_path=output_dir)
    
    print("Extracting audio...")
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(output_path=output_dir)
    
    return video_file, audio_file

def convert_to_wav(audio_file, output_dir):
    print("Converting to WAV...")
    audio = AudioSegment.from_file(audio_file)
    wav_file = os.path.join(output_dir, os.path.basename(audio_file).rsplit(".", 1)[0] + ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

def save_analysis(analysis, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

def print_analysis_preview(analysis):
    print("\nAnalysis Preview:")
    if isinstance(analysis, dict):
        for key, value in analysis.items():
            if isinstance(value, str):
                print(f"{key}: {value[:100]}...")
            elif isinstance(value, list):
                print(f"{key}: {', '.join(str(item) for item in value[:3])}...")
            else:
                print(f"{key}: {str(value)[:100]}...")
    else:
        print("Analysis is not in the expected format. Here's a preview of the raw content:")
        print(str(analysis)[:500] + "...")

def setup_whisper_pipeline():
    print("Setting up Whisper pipeline...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )

    return pipe

def process_audio(pipe, audio):
    print("Transcribing and translating audio to English...")
    result = pipe(audio, generate_kwargs={"task": "translate", "language": "en"})
    return result

def save_text(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(text, f, indent=2, ensure_ascii=False)

def summarize_transcript(transcription_result):
    print("Analyzing transcript...")
    
    # Prepare the transcript with timestamps, handling potential missing timestamps
    timestamped_transcript = []
    for chunk in transcription_result.get('chunks', []):
        start = format_timestamp(chunk['timestamp'][0] if chunk['timestamp'] and len(chunk['timestamp']) > 0 else None)
        end = format_timestamp(chunk['timestamp'][1] if chunk['timestamp'] and len(chunk['timestamp']) > 1 else None)
        timestamped_transcript.append(f"[{start}-{end}] {chunk['text']}")
    
    timestamped_transcript = "\n".join(timestamped_transcript)
    
    prompt = f"""Analyze the following transcript and provide a detailed report including:

1. Executive Summary (100-150 words): A concise overview of the main points and key takeaways.
2. Key Findings: List 7-10 important facts, insights, or conclusions from the content.
3. Significant Quotes: Extract 5-7 impactful quotes, each 2-4 sentences long. Include timestamps, your reaction, and their significance.
4. Critical Analysis: Provide a 200-300 word critical analysis of the content, discussing its strengths, weaknesses, and potential biases.
5. Contextual Background: Offer 150-200 words of relevant background information to help understand the context of the discussion.
6. Follow-up Questions: Suggest 8-12 questions for further exploration of the topic.
7. Key Vocabulary: Identify and define 8-10 important terms or concepts from the transcript.
8. Themes and Topics: Identify 5-7 main themes or topics discussed, with a brief explanation of each.
9. Stakeholder Analysis: Identify key stakeholders mentioned or implied in the content and their perspectives (150-200 words).
10. Implications and Applications: Discuss potential real-world implications or applications of the ideas presented (200-250 words).
11. Detailed Notes Outline: Create a structured outline of the content with main sections, subsections, and corresponding timestamps.
12. In-Depth Notes: Provide a detailed analysis of the content, organized by the outline structure. Include key points, arguments, supporting evidence, examples, and any statistical data mentioned. This should be a comprehensive breakdown of the entire discussion.

Here's the timestamped transcript:

{timestamped_transcript}

Respond only with the JSON object, no additional text. Use the following structure:

{{
  "executiveSummary": "Summary text here...",
  "keyFindings": ["Finding 1", "Finding 2", ...],
  "significantQuotes": [
    {{
      "quote": "Quote text here...",
      "reaction": "Your reaction to the quote...",
      "significance": "Why this quote is important...",
      "timestamp": [start_time, end_time]
    }},
    ...
  ],
  "criticalAnalysis": "Critical analysis text here...",
  "contextualBackground": "Background information text here...",
  "followUpQuestions": ["Question 1?", "Question 2?", ...],
  "keyVocabulary": [
    {{
      "term": "Term 1",
      "definition": "Definition of term 1"
    }},
    ...
  ],
  "themesAndTopics": [
    {{
      "theme": "Theme 1",
      "explanation": "Brief explanation of theme 1"
    }},
    ...
  ],
  "stakeholderAnalysis": "Stakeholder analysis text here...",
  "implicationsAndApplications": "Implications and applications text here...",
  "notesOutline": [
    {{
      "title": "Main Section 1",
      "timestamp": [start_time, end_time],
      "subsections": [
        {{
          "title": "Subsection 1.1",
          "timestamp": [start_time, end_time]
        }},
        ...
      ]
    }},
    ...
  ],
  "detailedNotes": [
    {{
      "section": "Main Section 1",
      "content": "Detailed notes for Main Section 1...",
      "subsections": [
        {{
          "title": "Subsection 1.1",
          "content": "Detailed notes for Subsection 1.1..."
        }},
        ...
      ]
    }},
    ...
  ]
}}

"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        analysis = json.loads(response.content[0].text)
        return analysis
    except json.JSONDecodeError:
        print("Error parsing JSON response. Returning raw text.")
        return response.content[0].text

def process_video(url):
    try:
        yt = YouTube(url)
        output_dir = create_output_directory(yt.video_id)
        
        _, audio_file = download_video(url, output_dir)
        wav_file = convert_to_wav(audio_file, output_dir)
        
        pipe = setup_whisper_pipeline()
        
        # Transcription and Translation to English
        transcription_result = process_audio(pipe, wav_file)
        if transcription_result:
            transcript_file = os.path.join(output_dir, "english_transcription.json")
            save_text(transcription_result, transcript_file)
            print(f"English transcription with timestamps saved to {transcript_file}")
            
            # Enhanced Analysis with timestamps
            analysis = summarize_transcript(transcription_result)
            analysis_file = os.path.join(output_dir, "detailed_analysis.json")
            save_analysis(analysis, analysis_file)
            print(f"Detailed analysis saved to {analysis_file}")
            
            # Create quote snippets
            create_quote_snippets(audio_file, analysis, output_dir)
        else:
            print("Transcription and translation failed.")
        
        print("\nFiles saved:")
        print(f"Original Audio: {audio_file}")
        print(f"WAV Audio: {wav_file}")
        print(f"English Transcription with Timestamps: {transcript_file}")
        print(f"Detailed Analysis: {analysis_file}")
        print(f"Quote snippets have been saved in the output directory.")
        
        return output_dir, transcription_result, analysis
    except Exception as e:
        print(f"An error occurred during video processing: {str(e)}")
        return None, None, None

def create_quote_snippets(audio_file, analysis, output_dir):
    print("Creating quote snippets...")
    
    quotes = analysis.get('reactionQuotes', [])
    if not isinstance(quotes, list):
        print("reactionQuotes is not in the expected format.")
        return
    
    if not quotes:
        print("No quotes found in the analysis.")
        return
    
    try:
        # Load the entire audio file
        audio = AudioSegment.from_file(audio_file)
        
        for i, quote_obj in enumerate(quotes):
            try:
                timestamp_og = quote_obj.get('timestamp')
                if not timestamp_og or len(timestamp_og) != 2:
                    print(f"Invalid timestamp for quote: {quote_obj.get('quote', '')}")
                    continue

                audio_length_sec = len(audio) / 1000  # Convert milliseconds to seconds

                # Add five seconds to either side of the timestamp, clamping to audio bounds
                start_time = max(0, timestamp_og[0] - 2)
                end_time = min(audio_length_sec, timestamp_og[1] + 10)

                # Convert timestamp to milliseconds
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                # Extract the audio snippet
                audio_snippet = audio[start_ms:end_ms]
                
                # Save the audio snippet
                audio_snippet_file = os.path.join(output_dir, f"quote_audio_{i+1}.mp3")
                audio_snippet.export(audio_snippet_file, format="mp3")
                print(f"Audio snippet saved to {audio_snippet_file}")
            
            except Exception as e:
                print(f"Error processing quote: {quote_obj.get('quote', '')}")
                print(f"Error details: {str(e)}")
                continue
        
    except Exception as e:
        print(f"An error occurred while creating quote snippets: {str(e)}")
if __name__ == "__main__":
    video_url = input("Enter the YouTube video URL: ")
    result = process_video(video_url)
    
    if result[0] is not None:
        output_dir, transcription, analysis = result
        print(f"\nAll files have been saved in the directory: {output_dir}")
        
        if analysis:
            print_analysis_preview(analysis)
            print("\nFull Analysis Structure:")
            print(json.dumps(analysis, indent=2))
        else:
            print("Analysis could not be generated.")
    else:
        print("Video processing failed. Please check the error messages above.")  
    