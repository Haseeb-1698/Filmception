import os
import re
import time
import platform
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor

# Try to import playsound with fallback
try:
    from playsound import playsound
except ImportError:
    # Define fallback for environments without playsound
    def playsound(file):
        print(f"Would play sound file: {file} (playsound not available)")

class AudioConverter:
    """Service for converting text to speech audio with parallel processing"""
    
    def __init__(self, output_dir="audio_summaries", max_workers=4):
        """Initialize audio conversion service.
        
        Args:
            output_dir: Directory to store audio files
            max_workers: Maximum number of parallel workers for batch processing
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        os.makedirs(output_dir, exist_ok=True)
    
    def prepare_text_for_tts(self, text):
        """Prepare text for better TTS quality.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return "."
        
        # Remove consecutive duplicate words
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
        
        # Add periods if missing to create natural pauses
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Expand common abbreviations
        text = text.replace(" vs ", " versus ")
        
        return text
    
    def create_audio(self, text, lang, output_file=None):
        """Convert text to audio and save as MP3.
        
        Args:
            text: Text to convert
            lang: Language code
            output_file: Path to save MP3 (optional)
            
        Returns:
            tuple: (success status, output file path)
        """
        if not output_file:
            import uuid
            file_id = str(uuid.uuid4())[:8]
            output_file = os.path.join(self.output_dir, f"audio_{file_id}_{lang}.mp3")
        
        try:
            # Prepare text for TTS
            prepared_text = self.prepare_text_for_tts(text)
            
            # Print the prepared text (required line preserved)
            movie_id = os.path.basename(output_file).split('_')[0] if '_' in os.path.basename(output_file) else 'unknown'
            print(f"Prepared text for {movie_id} ({lang}): {prepared_text}")
            
            # Generate audio
            tts = gTTS(text=prepared_text, lang=lang, slow=False)
            tts.save(output_file)
            
            print(f"Audio file created: {output_file}")
            return True, output_file
            
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return False, None
    
    def play_audio(self, audio_file):
        """Play audio file with robust error handling.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            bool: Success status
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return False
            
        try:
            # Try using playsound first
            playsound(audio_file)
            return True
        except Exception as e:
            print(f"Error with playsound: {e}")
            
            # Fall back to system commands
            try:
                system = platform.system()
                
                if system == 'Windows':
                    os.system(f'start "{audio_file}"')  # Added quotes to handle paths with spaces
                elif system == 'Darwin':  # macOS
                    os.system(f'afplay "{audio_file}"')
                elif system == 'Linux':
                    os.system(f'xdg-open "{audio_file}"')
                    
                print(f"Attempted to play using system command")
                return True
            except Exception as e2:
                print(f"Error playing audio with system command: {e2}")
                return False
            
    def batch_create_audio(self, tasks):
        """Create multiple audio files in parallel.
        
        Args:
            tasks: List of tuples (text, lang, output_file)
            
        Returns:
            list: List of (success status, output file path) tuples
        """
        def process_task(task):
            text, lang, output_file = task
            return self.create_audio(text, lang, output_file)
            
        print(f"Processing {len(tasks)} audio tasks using {self.max_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_task, tasks))
            
        success_count = sum(1 for r in results if r[0])
        print(f"Successfully processed {success_count}/{len(tasks)} audio files")
        return results 