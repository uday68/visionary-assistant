import RPi.GPIO as gp
import time  # Don't forget to import time for sleep functionality

class ServoPositionHandler:
    def __init__(self):
        # Set up GPIO mode
        gp.setmode(gp.BOARD)
        
        # Define the initial position and the servo pins (only two servos)
        self.position = 0
        self.servo_pins = [7, 11]  # Change to only two pins
        
        # Set up servo pins as output
        for pin in self.servo_pins:
            gp.setup(pin, gp.OUT)
        
        # Create PWM instances for each servo pin
        self.pwms = [gp.PWM(pin, 50) for pin in self.servo_pins]
        for pwm in self.pwms:
            pwm.start(0)  # Initial duty cycle at 0%

        # Set initial servo angle (example: both servos at 90 degrees)
        self.set_angle([90, 90])

    def turn_angle(self, pwm, angle):
        """Turn the servo to a specified angle."""
        duty = angle / 18 + 2  # Convert angle to duty cycle
        self.position = duty
        pwm.ChangeDutyCycle(duty)  # Adjust servo angle
        time.sleep(1)  # Wait for servo to reach position
        pwm.ChangeDutyCycle(0)  # Stop sending signal to servo to prevent jitter

    def set_angle(self, angles):
        """Set multiple servos to specified angles (for two servos)."""
        try:
            for pwm, angle in zip(self.pwms, angles):
                self.turn_angle(pwm, angle)
                time.sleep(1)  # Delay between movements for stability
        except Exception as e:
            print(f"Error in setting angles: {e}")
            self.cleanup()

    def cleanup(self):
        """Clean up PWM and GPIO resources."""
        for pwm in self.pwms:
            pwm.stop()
        gp.cleanup()

    def sleep(self):
        """Put all servos to rest position."""
        self.set_angle([0, 0])  # Set both servos to 0 degrees

# Example usage:
if __name__ == "__main__":
    servo_handler = ServoPositionHandler()
    servo_handler.set_angle([90, 90])  # Example: Set both servos to 90 degrees
    time.sleep(2)  # Wait before putting them to rest position
    servo_handler.sleep()  # Set both servos to 0 degrees
