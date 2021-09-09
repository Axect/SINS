#[macro_use]
extern crate peroxide;
use peroxide::fuga::*;

use std::f64::consts::PI;
use std::thread;
use std::sync::mpsc;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let size: usize = args[1].parse().unwrap();
    //let (tx, rx) = mpsc::sync_channel(size);

    let u_r = Uniform(1f64, 2f64);
    let u_th = Uniform(0f64, 2f64*PI);
    let u_vx = Uniform(-0.2, -0.1);
    let u_vy = Uniform(0.1, 0.2);

    let r = u_r.sample(size);
    let th = u_th.sample(size);
    let x = r.mul_v(&th.fmap(|t| t.cos()));
    let y = r.mul_v(&th.fmap(|t| t.sin()));
    let vx = u_vx.sample(size);
    let vy = u_vy.sample(size);

    let z = hstack!(x, y, vx, vy);
    
    for i in 0 .. size {
        //let tx_sender = mpsc::SyncSender::clone(&tx);
        let z_sample = z.row(i);
        z_sample.print();
        //thread::spawn(move || {
            let init_state = State::<f64>::new(0f64, z_sample, c!(0,0,0,0));

            let mut ode_solver = ExplicitODE::new(eom);

            ode_solver
                .set_method(ExMethod::RK4)
                .set_initial_condition(init_state)
                .set_step_size(0.1)
                .set_times(100);

            let result = ode_solver.integrate();
            result.print();
            
            let mut df = DataFrame::new(vec![]);
            df.push("t", Series::new(result.col(0)));
            df.push("x", Series::new(result.col(1)));
            df.push("y", Series::new(result.col(2)));
            df.push("vx", Series::new(result.col(3)));
            df.push("vy", Series::new(result.col(4)));

            df.print();
        //});
    }
}

fn eom(st: &mut State<f64>, _: &NoEnv) {
    let dy = &mut st.deriv;
    let y = &st.value;
    dy[0] = y[2];
    dy[1] = y[3];
    let r = (y[0].powi(2) + y[1].powi(2)).sqrt().powi(3);
    dy[2] = - y[0] / r;
    dy[3] = - y[1] / r;
}
