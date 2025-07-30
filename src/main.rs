use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
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

        let magnitudes: Vec<f32> = output.iter().map(|c| c.norm()).collect();

        println!(
            "FFT Magnitudes (first 20): {:?}",
            &magnitudes[..20.min(magnitudes.len())]
        );
    }

    Ok(())
}
