use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use plotters::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use std::{
    error::Error,
    sync::{Arc, Mutex},
};

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

    if buffer.len() > 0 {
        let n = buffer.len().next_power_of_two();
        let mut input: Vec<Complex<f32>> = buffer
            .iter()
            .cloned()
            .map(|x| Complex::new(x, 0.0))
            .collect();
        input.resize(n, Complex::new(0.0, 0.0));

        let mut output = input.clone();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut output);

        let half_n = n / 2;
        let magnitudes: Vec<f32> = output[..half_n].iter().map(|c| c.norm()).collect();

        let freqs: Vec<f32> = (0..half_n)
            .map(|i| i as f32 * sample_rate as f32 / n as f32)
            .collect();

        plot_spectrum(&freqs, &magnitudes, "specturm.png")?;
        plot_waveform(&buffer, sample_rate, "waveform.png")?;
    }

    write_wav("test.wav", &buffer, sample_rate)?;

    Ok(())
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

fn write_wav(
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

fn plot_spectrum(freqs: &[f32], magnitudes: &[f32], filename: &str) -> Result<(), Box<dyn Error>> {
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

fn plot_waveform(
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
