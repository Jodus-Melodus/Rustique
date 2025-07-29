use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
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
    }

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
