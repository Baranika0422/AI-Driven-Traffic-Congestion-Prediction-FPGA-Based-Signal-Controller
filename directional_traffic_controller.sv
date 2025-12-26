
module directional_traffic_controller (
    input wire clk, reset,
    input wire [1:0] north_ml_level, south_ml_level,
    input wire [1:0] east_ml_level, west_ml_level,
    input wire ml_prediction_valid,
    input wire emergency_override,
    input wire [1:0] emergency_direction,
    input wire [7:0] north_queue, south_queue,
    input wire [7:0] east_queue, west_queue,
    output reg [2:0] north_light, south_light, east_light, west_light,
    output reg [3:0] current_phase,
    output reg [31:0] phase_timer,  // Changed from [15:0] to [31:0] for decimal
    output reg [1:0] current_mode
);

    localparam ONE_SECOND = 50;
    localparam RED = 3'b100, YELLOW = 3'b010, GREEN = 3'b001;
    
    localparam PHASE_NS_GREEN = 4'd0;
    localparam PHASE_NS_YELLOW = 4'd1;
    localparam PHASE_ALL_RED_1 = 4'd2;
    localparam PHASE_EW_GREEN = 4'd3;
    localparam PHASE_EW_YELLOW = 4'd4;
    localparam PHASE_ALL_RED_2 = 4'd5;
    
    localparam MODE_NORMAL = 2'b00;
    localparam MODE_ADAPTIVE = 2'b01;
    localparam MODE_EMERGENCY = 2'b10;
    
    reg [1:0] stored_north_level, stored_south_level;
    reg [1:0] stored_east_level, stored_west_level;
    
    reg [31:0] green_time_ns, green_time_ew;
    reg [31:0] yellow_time, all_red_time;
    reg [31:0] cycle_counter;
    reg [31:0] phase_duration;
    reg adaptive_mode_active;
    
    // Calculate phase durations based on ML predictions and queue lengths
    always @(*) begin
        yellow_time = 3 * ONE_SECOND;
        all_red_time = 2 * ONE_SECOND;
        
        if (adaptive_mode_active) begin
            case (stored_north_level > stored_south_level ? stored_north_level : stored_south_level)
                2'b00: green_time_ns = 30 * ONE_SECOND;
                2'b01: green_time_ns = 40 * ONE_SECOND;
                2'b10: green_time_ns = 50 * ONE_SECOND;
                2'b11: green_time_ns = 60 * ONE_SECOND;
            endcase
            
            case (stored_east_level > stored_west_level ? stored_east_level : stored_west_level)
                2'b00: green_time_ew = 30 * ONE_SECOND;
                2'b01: green_time_ew = 40 * ONE_SECOND;
                2'b10: green_time_ew = 50 * ONE_SECOND;
                2'b11: green_time_ew = 60 * ONE_SECOND;
            endcase
            
            if (north_queue > south_queue + 20 || south_queue > north_queue + 20)
                green_time_ns = green_time_ns + (10 * ONE_SECOND);
                
            if (east_queue > west_queue + 20 || west_queue > east_queue + 20)
                green_time_ew = green_time_ew + (10 * ONE_SECOND);
        end else begin
            green_time_ns = 20 * ONE_SECOND;
            green_time_ew = 20 * ONE_SECOND;
        end
        
        case(current_phase)
            PHASE_NS_GREEN: phase_duration = green_time_ns;
            PHASE_NS_YELLOW: phase_duration = yellow_time;
            PHASE_ALL_RED_1: phase_duration = all_red_time;
            PHASE_EW_GREEN: phase_duration = green_time_ew;
            PHASE_EW_YELLOW: phase_duration = yellow_time;
            PHASE_ALL_RED_2: phase_duration = all_red_time;
            default: phase_duration = green_time_ns;
        endcase
    end
    
    // Store ML predictions
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            stored_north_level <= 2'b00;
            stored_south_level <= 2'b00;
            stored_east_level <= 2'b00;
            stored_west_level <= 2'b00;
            adaptive_mode_active <= 0;
        end else begin
            if (ml_prediction_valid) begin
                stored_north_level <= north_ml_level;
                stored_south_level <= south_ml_level;
                stored_east_level <= east_ml_level;
                stored_west_level <= west_ml_level;
                adaptive_mode_active <= 1;
            end
        end
    end
    
    // Main traffic light state machine
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            current_phase <= PHASE_NS_GREEN;
            cycle_counter <= 0;
            phase_timer <= 30;
            current_mode <= MODE_NORMAL;
            north_light <= GREEN;
            south_light <= GREEN;
            east_light <= RED;
            west_light <= RED;
        end else begin
            if (emergency_override) begin
                current_mode <= MODE_EMERGENCY;
                cycle_counter <= 0;
                phase_timer <= 999;
                
                // FIXED: Only the specific emergency direction gets GREEN, all others RED
                case (emergency_direction)
                    2'b00: begin  // NORTH emergency - ONLY North green
                        north_light <= GREEN;
                        south_light <= RED;
                        east_light <= RED;
                        west_light <= RED;
                    end
                    2'b01: begin  // SOUTH emergency - ONLY South green
                        north_light <= RED;
                        south_light <= GREEN;
                        east_light <= RED;
                        west_light <= RED;
                    end
                    2'b10: begin  // EAST emergency - ONLY East green
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= GREEN;
                        west_light <= RED;
                    end
                    2'b11: begin  // WEST emergency - ONLY West green
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= RED;
                        west_light <= GREEN;
                    end
                endcase
            end else begin
                current_mode <= adaptive_mode_active ? MODE_ADAPTIVE : MODE_NORMAL;
                
                // Calculate decimal timer value
                if (cycle_counter == 0) begin
                    phase_timer <= phase_duration / ONE_SECOND;
                end else begin
                    phase_timer <= (phase_duration - cycle_counter) / ONE_SECOND;
                end
                
                cycle_counter <= cycle_counter + 1;
                
                case (current_phase)
                    PHASE_NS_GREEN: begin
                        north_light <= GREEN;
                        south_light <= GREEN;
                        east_light <= RED;
                        west_light <= RED;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_NS_YELLOW;
                            cycle_counter <= 0;
                        end
                    end
                    
                    PHASE_NS_YELLOW: begin
                        north_light <= YELLOW;
                        south_light <= YELLOW;
                        east_light <= RED;
                        west_light <= RED;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_ALL_RED_1;
                            cycle_counter <= 0;
                        end
                    end
                    
                    PHASE_ALL_RED_1: begin
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= RED;
                        west_light <= RED;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_EW_GREEN;
                            cycle_counter <= 0;
                        end
                    end
                    
                    PHASE_EW_GREEN: begin
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= GREEN;
                        west_light <= GREEN;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_EW_YELLOW;
                            cycle_counter <= 0;
                        end
                    end
                    
                    PHASE_EW_YELLOW: begin
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= YELLOW;
                        west_light <= YELLOW;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_ALL_RED_2;
                            cycle_counter <= 0;
                        end
                    end
                    
                    PHASE_ALL_RED_2: begin
                        north_light <= RED;
                        south_light <= RED;
                        east_light <= RED;
                        west_light <= RED;
                        
                        if (cycle_counter >= phase_duration) begin
                            current_phase <= PHASE_NS_GREEN;
                            cycle_counter <= 0;
                        end
                    end
                    
                    default: begin
                        current_phase <= PHASE_NS_GREEN;
                        cycle_counter <= 0;
                    end
                endcase
            end
        end
    end
    
endmodule