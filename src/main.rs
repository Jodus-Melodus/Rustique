use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use plotters::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex32};
use std::{
    error::Error,
    f32::consts::PI,
    sync::{Arc, Mutex},
    thread::sleep,
    time::Duration,
};

struct Rustique {
    detected_note: Arc<Mutex<String>>,
    detected_freq: Arc<Mutex<f32>>,
}

impl eframe::App for Rustique {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let note = self.detected_note.lock().unwrap().clone();
        let freq = *self.detected_freq.lock().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Rustique Tuner");
            ui.label(format!("Detected note: {}", note));
            ui.label(format!("Frequency: {:.2} Hz", freq));
        });
    }
}

static NOTES: [(&str, f32); 12] = [
    ("C", 261.63),
    ("C#", 277.18),
    ("D", 293.66),
    ("D#", 311.13),
    ("E", 329.63),
    ("F", 349.23),
    ("F#", 369.99),
    ("G", 392.00),
    ("G#", 415.30),
    ("A", 440.00),
    ("A#", 466.16),
    ("B", 493.88),
];

fn main() -> Result<(), Box<dyn Error>> {
    let detected_note = Arc::new(Mutex::new("A4".to_string()));
    let detected_freq = Arc::new(Mutex::new(440.0_f32));
    let note_clone = detected_note.clone();
    let freq_clone = detected_freq.clone();
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0 as usize;
    let window_size = 4096;
    let hop_size = window_size / 2;
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

    std::thread::spawn(move || {
        loop {
            sleep(Duration::from_millis(10));
            let mut buffer = match audio_data.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if buffer.len() < window_size {
                continue;
            }

            let stft_frames = compute_short_time_fourier_transform(&buffer, window_size, hop_size);
            if stft_frames.is_empty() {
                let drain_len = hop_size.min(buffer.len());
                buffer.drain(..drain_len);
                continue;
            }
            let frequency_magnitudes = stft_frames
                .iter()
                .map(|frame| {
                    frame[..window_size / 2]
                        .iter()
                        .map(|v| v.norm())
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<Vec<f32>>>();
            if frequency_magnitudes.is_empty() || frequency_magnitudes[0].is_empty() {
                let drain_len = hop_size.min(buffer.len());
                buffer.drain(..drain_len);
                continue;
            }

            let num_bins = frequency_magnitudes[0].len();
            let num_frames = frequency_magnitudes.len();
            let mut average_magnitudes_per_bin = vec![0.0f32; num_bins];
            for frame in &frequency_magnitudes {
                for (bin_idx, mag) in frame.iter().enumerate() {
                    average_magnitudes_per_bin[bin_idx] += *mag;
                }
            }
            for mag in &mut average_magnitudes_per_bin {
                *mag /= num_frames as f32;
            }

            if let Some((strongest_bin_idx, _)) = average_magnitudes_per_bin
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let freq_resolution = sample_rate as f32 / window_size as f32;
                let dominant_freq = strongest_bin_idx as f32 * freq_resolution;

                if let Some((note_name, note_freq)) = frequency_to_note(dominant_freq) {
                    *note_clone.lock().unwrap() = note_name.clone();
                    *freq_clone.lock().unwrap() = dominant_freq;
                    println!(
                        "Detected note: {} ({:.2} Hz), Detected freq: {:.2} Hz",
                        note_name, note_freq, dominant_freq
                    );
                }
            }

            let drain_len = hop_size.min(buffer.len());
            buffer.drain(..drain_len);
        }
    });

    let app = Rustique {
        detected_note,
        detected_freq,
    };
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Rustique Tuner",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    )?;
    Ok(())
}

fn frequency_to_note(freq: f32) -> Option<(String, f32)> {
    if freq <= 0.0 {
        return None;
    }
    let mut closest_note = None;
    let mut min_diff = f32::MAX;
    let mut closest_octave = 0;
    for octave in 0..8 {
        for (name, base_freq) in NOTES.iter() {
            let note_freq = base_freq * 2f32.powi(octave - 4);
            let diff = (freq - note_freq).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_note = Some((name, note_freq));
                closest_octave = octave;
            }
        }
    }
    closest_note.map(|(name, note_freq)| (format!("{}{}", name, closest_octave), note_freq))
}

fn _plot_average_magnitudes_with_bins(
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
        .x_labels(20)
        .x_label_formatter(&|x| format!("{:.1}", x))
        .draw()?;

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

fn _compute_bin_ranges(sample_rate: usize, window_size: usize) -> Vec<(f32, f32)> {
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
