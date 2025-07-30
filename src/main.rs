use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use plotters::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::{
    error::Error,
    f32::consts::PI,
    sync::{Arc, Mutex},
};

static NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

fn main() -> Result<(), Box<dyn Error>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    println!("Using input device: {}", device.name()?);

    let config = device.default_input_config()?;
    println!("Default input config: {:?}", config);

    let sample_rate = config.sample_rate().0 as usize;

    let audio_data = Arc::new(Mutex::new(Vec::<f32>::new()));

    let audio_data_clone = audio_data.clone();
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _| {
            let mut buffer = audio_data_clone.lock().unwrap();
            buffer.extend_from_slice(data);
        },
        move |err| eprintln!("Stream error: {:?}", err),
        None,
    )?;

    stream.play()?;
    println!("Recording... Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let buffer = audio_data.lock().unwrap();
    println!("Recorded {} samples", buffer.len());

    let window_size = 1024;
    let hop_size = window_size / 2;

    let mut frequencies = Vec::new();

    if buffer.len() >= window_size {
        let stft_frames = compute_short_time_fourier_transform(&buffer, window_size, hop_size);

        let frame_frequencies: Vec<f32> = (0..window_size / 2)
            .map(|i| i as f32 * sample_rate as f32 / window_size as f32)
            .collect();

        for _ in 0..stft_frames.len() {
            frequencies.push(frame_frequencies.clone());
        }

        // bin ranges and frequency magnitudes per bin per frame
        let bin_ranges = compute_bin_ranges(sample_rate, window_size);
        let frequency_magnitudes = stft_frames
            .iter()
            .map(|frame| {
                frame[..window_size / 2]
                    .iter()
                    .map(|v| v.norm())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        // average magnitudes per bin
        let num_bins = frequency_magnitudes[0].len();
        let num_frames = frequency_magnitudes.len();

        let bin_centers: Vec<f32> = bin_ranges
            .iter()
            .map(|(low, high)| (low + high) / 2.0)
            .collect();

        let mut average_magnitudes_per_bin = vec![0.0f32; num_bins];
        for frame in &frequency_magnitudes {
            for (bin_idx, mag) in frame.iter().enumerate() {
                average_magnitudes_per_bin[bin_idx] += *mag;
            }
        }

        for mag in &mut average_magnitudes_per_bin {
            *mag /= num_frames as f32;
        }

        plot_average_magnitudes_with_bins(&average_magnitudes_per_bin, &bin_centers)?;
    }

    Ok(())
}

fn frequency_to_note(frequency: f32) -> String {
    NOTES[(69.0 + 12.0 * (frequency / 440.0).log2()) as usize].to_string()
}

fn plot_average_magnitudes_with_bins(
    average_magnitudes: &[f32],
    bin_centers: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("average_magnitudes_bins.png", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_magnitude = average_magnitudes.iter().cloned().fold(f32::MIN, f32::max);

    // Define x range from min to max bin center
    let x_min = *bin_centers.first().unwrap_or(&0.0);
    let x_max = *bin_centers.last().unwrap_or(&0.0);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Average Magnitudes per Frequency Bin",
            ("sans-serif", 30).into_font(),
        )
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0f32..max_magnitude)?;

    // Customize x axis to show bin centers with formatted labels
    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Average Magnitude")
        .x_labels(10)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

    // Draw points
    chart.draw_series(
        average_magnitudes
            .iter()
            .zip(bin_centers.iter())
            .map(|(mag, center)| Circle::new((*center, *mag), 3, RED.filled())),
    )?;

    // Connect points with line
    chart.draw_series(LineSeries::new(
        bin_centers
            .iter()
            .zip(average_magnitudes.iter())
            .map(|(center, mag)| (*center, *mag)),
        &BLUE,
    ))?;

    Ok(())
}

fn compute_bin_ranges(sample_rate: usize, window_size: usize) -> Vec<(f32, f32)> {
    let bin_width = sample_rate as f32 / window_size as f32;
    let half_n = window_size / 2;
    (0..half_n)
        .map(|i| {
            let center = i as f32 * bin_width;
            (center - bin_width / 2.0, center + bin_width / 2.0)
        })
        .collect()
}

fn compute_short_time_fourier_transform(
    buffer: &[f32],
    window_size: usize,
    hop_size: usize,
) -> Vec<Vec<Complex32>> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);
    let hann: Vec<f32> = (0..window_size)
        .map(|i| (PI * 2.0 * i as f32 / window_size as f32).sin().powi(2))
        .collect();
    let mut spectrum = Vec::new();
    let mut pos = 0;

    while pos + window_size <= buffer.len() {
        let mut windowed: Vec<Complex32> = buffer[pos..pos + window_size]
            .iter()
            .zip(hann.iter())
            .map(|(sample, w)| Complex32::new(sample * w, 0.0))
            .collect();

        fft.process(&mut windowed);
        spectrum.push(windowed);
        pos += hop_size;
    }

    spectrum
}

fn _read_wav(path: &str) -> Result<(usize, Vec<f32>), Box<dyn Error>> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.map_err(|e| e.into()))
            .collect::<Result<_, Box<dyn Error>>>()?,
        SampleFormat::Int => {
            let max_amplitude = 2_i32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| Ok(s? as f32 / max_amplitude))
                .collect::<Result<_, Box<dyn Error>>>()?
        }
    };

    Ok((spec.sample_rate as usize, samples))
}

fn _write_wav(
    path: &str,
    samples: &[f32],
    sample_rate: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sample_rate as u32,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample.clamp(-1.0, 1.0))?;
    }
    writer.finalize()?;
    Ok(())
}

fn _plot_spectrum(freqs: &[f32], magnitudes: &[f32], filename: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_y = magnitudes.iter().copied().fold(0.0_f32, f32::max);
    let x_max = freqs.last().copied().unwrap_or(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Frequency Specturm", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0_f32..x_max, 0.0_f32..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Magnitude")
        .draw()?;

    chart.draw_series(LineSeries::new(
        freqs.iter().zip(magnitudes.iter()).map(|(&x, &y)| (x, y)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}

fn _plot_waveform(
    samples: &[f32],
    sample_rate: usize,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(filename, (1024, 512)).into_drawing_area();
    root.fill(&WHITE)?;

    // Determine time range in seconds
    let duration = samples.len() as f32 / sample_rate as f32;

    // Find the min/max amplitude for Y-axis scaling
    let (min_y, max_y) = samples
        .iter()
        .fold((f32::MAX, f32::MIN), |(min, max), &val| {
            (min.min(val), max.max(val))
        });

    let mut chart = ChartBuilder::on(&root)
        .caption("Audio Waveform", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..duration, min_y..max_y)?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Amplitude")
        .x_labels(10)
        .y_labels(5)
        .draw()?;

    chart.draw_series(LineSeries::new(
        samples
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f32 / sample_rate as f32, y)),
        &BLUE,
    ))?;

    root.present()?;
    Ok(())
}
