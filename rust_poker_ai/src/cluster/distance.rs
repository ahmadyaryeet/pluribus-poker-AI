use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, Child, ChildStdin, ChildStdout};
use std::io::{Write, BufReader, BufRead};

pub struct EmdCalculator {
    child: Child,
    child_stdin: ChildStdin,
    child_stdout: BufReader<ChildStdout>,
}

impl EmdCalculator {
    // Constructor to initialize the EmdCalculator
    pub fn new() -> Self {
        // Get the current directory
        let current_dir = env::current_dir().expect("Failed to get current directory");
        println!("Current directory: {:?}", current_dir);

        // Find the Cargo.toml file by going up in the directory hierarchy
        let mut cargo_toml_dir = current_dir.clone();
        loop {
            let cargo_toml_path = cargo_toml_dir.join("Cargo.toml");
            if cargo_toml_path.is_file() {
                break;
            }
            if !cargo_toml_dir.pop() {
                panic!("Cannot find Cargo.toml");
            }
        }

        println!("Cargo.toml directory: {:?}", cargo_toml_dir);

        let calc_emd_path = cargo_toml_dir.join("target/release/calc_emd");
        println!("calc_emd path: {:?}", calc_emd_path);

        if !calc_emd_path.exists() {
            panic!("calc_emd executable does not exist at {:?}", calc_emd_path);
        }

        if !calc_emd_path.is_file() {
            panic!("calc_emd path is not a file: {:?}", calc_emd_path);
        }

        // Debugging permission issues
        println!("Attempting to start calc_emd process...");

        let mut child = Command::new(&calc_emd_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect(&format!("Failed to start process: {:?}", calc_emd_path));

        println!("calc_emd process started successfully.");

        let child_stdin = child.stdin.take().expect("Failed to take child stdin");
        let child_stdout = BufReader::new(child.stdout.take().expect("Failed to take child stdout"));

        Self { child, child_stdin, child_stdout }
    }

    // Method to calculate the EMD distance between two vectors
    pub fn calc(&mut self, a: &Vec<f64>, b: &Vec<f64>) -> Result<f64, Box<dyn std::error::Error>> {
        let input = format!(
            "{};{}\n",
            a.iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(","),
            b.iter().map(|&num| num.to_string()).collect::<Vec<String>>().join(",")
        );

        println!("Input to calc_emd: {}", input);

        self.child_stdin.write_all(input.as_bytes())?;
        let mut output = String::new();
        self.child_stdout.read_line(&mut output)?;
        println!("Output from calc_emd: {}", output);
        output.trim().parse::<f64>().map_err(|e| e.into())
    }

    // Method to close the EmdCalculator and terminate the child process
    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.child.kill()?;
        self.child.wait()?;
        Ok(())
    }
}

impl Drop for EmdCalculator {
    // Automatically close the EmdCalculator when it goes out of scope
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn main() {
    println!("Starting EmdCalculator...");
    let mut calculator = EmdCalculator::new();
    let vec_a = vec![1.0, 2.0, 3.0];
    let vec_b = vec![4.0, 5.0, 6.0];
    match calculator.calc(&vec_a, &vec_b) {
        Ok(result) => println!("EMD result: {}", result),
        Err(e) => println!("Error calculating EMD: {}", e),
    }
    println!("EmdCalculator finished.");
}
