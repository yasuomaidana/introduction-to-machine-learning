use nannou::prelude::*;
use nannou_egui::{self, egui, Egui};

fn main() {
    nannou::app(model).update(update).run();
}

struct Settings {
    rotation: f32,
    color: Srgb<u8>,
    position: Vec2,
}

struct Model {
    settings: Settings,
    egui: Egui,
}

fn model(app: &App) -> Model {
    // Create window
    let window_id = app
        .new_window()
        .view(view)
        .raw_event(raw_window_event)
        .build()
        .unwrap();
    let window = app.window(window_id).unwrap();

    let egui = Egui::from_window(&window);

    Model {
        egui,
        settings: Settings {
            rotation: 0.0,
            color: WHITE,
            position: vec2(0.0, 0.0),
        },
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {
    let egui = &mut model.egui;
    let settings = &mut model.settings;

    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();

    egui::Window::new("Settings").show(&ctx, |ui| {
        // Resolution slider
        ui.label("Rotation:");
        ui.add(egui::Slider::new(&mut settings.rotation, 0.0..=360.0));

        // // Scale slider
        // ui.label("Scale:");
        // ui.add(egui::Slider::new(&mut settings.scale, 0.0..=1000.0));

        // Rotation slider
        // ui.label("Rotation:");
        // ui.add(egui::Slider::new(&mut settings.rotation, 0.0..=360.0));

        // Random color button
        let clicked = ui.button("Random color").clicked();

        if clicked {
            settings.color = rgb(random(), random(), random());
        }
    });
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    // Let egui handle things like keyboard and mouse input.
    model.egui.handle_raw_event(event);
}

fn point_plus_offset(x: f32, y: f32, offset:&Point2) -> Point2 {

    pt2(x + offset.x, y + offset.y)
}
fn view(app: &App, model: &Model, frame: Frame) {
    let settings = &model.settings;

    let half_length = 15.0/2.0;
    let offset = Point2::new(-5.0 + settings.position.x, 5.0 + settings.position.y);


    let p1 = point_plus_offset(half_length, half_length, &offset);
    let p2 = point_plus_offset(half_length, -half_length, &offset);
    let p3 = point_plus_offset(-half_length, -half_length, &offset);
    let p4 = point_plus_offset(-half_length, half_length, &offset);

    let points = vec![p1, p2, p3, p4];



    let draw = app.draw();
    draw.background().color(BLACK);


    draw.polygon()
        .color(BLACK)
        .stroke(PINK)
        .stroke_weight(20.0)
        .join_round()
        .points(points);

    draw.ellipse()
        .color(settings.color)
        .w_h(10.0, 10.0)
        .x_y(settings.position.x, settings.position.y);


    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}