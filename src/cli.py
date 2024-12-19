import os
import click
import logging
import time
from typing import List

from config import config
from video_processor import VideoProcessor, VideoHandler
from watchdog.observers import Observer

@click.group()
def cli():
    """
    YouTube Video Processing Pipeline CLI
    """
    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

@cli.command()
@click.option('--input-dir', help='Input directory for videos')
@click.option('--output-dir', help='Output directory for processed videos')
def process(input_dir: str = None, output_dir: str = None):
    """
    Process videos in the specified directory
    """
    # Use provided dirs or fall back to config
    input_directory = input_dir or config.get('input_directory')
    output_directory = output_dir or config.get('output_directory')

    # Validate directories
    if not os.path.exists(input_directory):
        click.echo(f"Input directory does not exist: {input_directory}")
        return

    os.makedirs(output_directory, exist_ok=True)

    # Get Gemini API keys from config
    api_keys = config.get('ai.api_keys', [])
    if not api_keys:
        click.echo("No Gemini API keys found in configuration")
        return

    # Initialize video processor
    processor = VideoProcessor(input_directory, output_directory, api_keys)

    # Process all videos in input directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    processed_videos = 0

    for filename in os.listdir(input_directory):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_directory, filename)
            click.echo(f"Processing video: {filename}")
            
            try:
                processed_video = processor.process_video(video_path)
                if processed_video:
                    processed_videos += 1
                    click.echo(f"Processed: {processed_video}")
            except Exception as e:
                click.echo(f"Error processing {filename}: {e}")

    click.echo(f"Processed {processed_videos} videos")

@cli.command()
def watch():
    """
    Start watching input directory for new videos
    """
    # Get configuration
    input_directory = config.get('input_directory')
    output_directory = config.get('output_directory')

    # Validate directories
    if not os.path.exists(input_directory):
        click.echo(f"Input directory does not exist: {input_directory}")
        return

    os.makedirs(output_directory, exist_ok=True)

    # Get Gemini API keys from config
    api_keys = config.get('ai.api_keys', [])
    if not api_keys:
        click.echo("No Gemini API keys found in configuration")
        return

    # Initialize video processor
    processor = VideoProcessor(input_directory, output_directory, api_keys)
    
    # Setup file system watcher
    event_handler = VideoHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, input_directory, recursive=False)
    
    click.echo(f"Watching directory: {input_directory}")
    click.echo("Press Ctrl+C to stop")

    try:
        observer.start()
        
        # Keep main thread running
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        click.echo("\nStopping video processing pipeline...")
        observer.stop()
    except Exception as e:
        click.echo(f"Unexpected error: {e}")
    finally:
        observer.join()

@cli.command()
@click.option('--key', required=True, help='Configuration key to retrieve')
def get_config(key: str):
    """
    Retrieve a specific configuration value
    """
    value = config.get(key)
    if value is not None:
        click.echo(f"{key}: {value}")
    else:
        click.echo(f"Configuration key not found: {key}")

@cli.command()
@click.option('--input-dir', help='Input directory for videos')
@click.option('--output-dir', help='Output directory for processed videos')
@click.option('--config-file', default='config.yaml', help='Path to configuration file')
def generate_config(input_dir: str, output_dir: str, config_file: str):
    """
    Generate a new configuration file
    """
    # Create a new configuration dictionary
    new_config = {
        'input_directory': input_dir or './input',
        'output_directory': output_dir or './output',
        'logging': {
            'level': 'INFO',
            'file': 'video_pipeline.log'
        },
        'video_processing': {
            'target_resolution': [1280, 720],
            'crf': 23,
            'preset': 'medium'
        },
        'ai': {
            'api_keys': [],
            'rate_limit': {
                'requests_per_minute': 15,
                'total_requests_per_day': 1500
            }
        }
    }

    # Save the configuration
    try:
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
        click.echo(f"Configuration file generated: {config_file}")
    except Exception as e:
        click.echo(f"Error generating configuration: {e}")

@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
def analyze(video_path: str):
    """
    Analyze a single video and display detailed insights
    """
    from video_analysis import analyze_video
    
    # Perform video analysis
    video_profile = analyze_video(video_path)
    
    if video_profile:
        # Pretty print the analysis results
        click.echo("\n--- Video Analysis ---")
        click.echo(f"Filename: {video_profile['filename']}")
        click.echo(f"Duration: {video_profile['duration']:.2f} seconds")
        click.echo(f"Total Frames: {video_profile['total_frames']}")
        click.echo(f"FPS: {video_profile['fps']}")
        
        click.echo("\nScene Changes:")
        click.echo(f"Total Scene Changes: {len(video_profile['scene_changes'])}")
        
        click.echo("\nMotion Intensity:")
        click.echo(f"Mean: {video_profile['motion_intensity']['mean']:.2f}")
        click.echo(f"Max: {video_profile['motion_intensity']['max']:.2f}")
        click.echo(f"Min: {video_profile['motion_intensity']['min']:.2f}")
        
        click.echo("\nBrightness Profile:")
        click.echo(f"Mean: {video_profile['brightness_profile']['mean']:.2f}")
        click.echo(f"Max: {video_profile['brightness_profile']['max']:.2f}")
        click.echo(f"Min: {video_profile['brightness_profile']['min']:.2f}")
        
        click.echo("\nAnalysis Metrics:")
        for metric, value in video_profile['analysis_metrics'].items():
            click.echo(f"{metric.replace('_', ' ').title()}: {value}")
    else:
        click.echo("Video analysis failed.")

if __name__ == '__main__':
    cli()
