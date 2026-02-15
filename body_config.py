class WholeBodyConfig:
    def __init__(self):
        # 0-based index definition based on the provided image
        # Total points: 133

        # ---------------------------------------------------------
        # 1. BODY (17 points): Index 0-16
        # Matches COCO standard
        # ---------------------------------------------------------
        self.body_skeleton = [
            [15, 13],
            [13, 11],
            [16, 14],
            [14, 12],
            [11, 12],  # Legs & Hips
            [5, 11],
            [6, 12],  # Torso sides
            [5, 6],
            [5, 7],
            [6, 8],
            [8, 10],
            [7, 9],  # Shoulders & Arms
            [1, 2],
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],  # Head & Neck
        ]

        # ---------------------------------------------------------
        # 2. FEET (6 points): Index 17-22
        # Image indices 18-23 -> Code indices 17-22
        # ---------------------------------------------------------
        self.feet_skeleton = [
            [15, 17],
            [15, 18],
            [17, 18],  # Left Foot (Ankle to Toe/Heel)
            [16, 19],
            [16, 20],
            [19, 20],  # Right Foot
        ]

        # ---------------------------------------------------------
        # 3. FACE (68 points): Index 23-90
        # Image indices 24-91 -> Code indices 23-90
        # Uses standard 68-point landmarks topology
        # ---------------------------------------------------------
        self.face_skeleton = []
        # Contour (Jaw)
        for i in range(23, 39):
            self.face_skeleton.append([i, i + 1])
        # Eyebrows (Right & Left)
        for i in range(40, 44):
            self.face_skeleton.append([i, i + 1])
        for i in range(45, 49):
            self.face_skeleton.append([i, i + 1])
        # Nose (Bridge & Bottom)
        for i in range(50, 53):
            self.face_skeleton.append([i, i + 1])
        for i in range(54, 58):
            self.face_skeleton.append([i, i + 1])
        self.face_skeleton.append([53, 54])  # Connect bridge to nose tip
        # Eyes
        for i in range(59, 64):
            self.face_skeleton.append([i, i + 1])
        self.face_skeleton.append([64, 59])  # Close Left Eye loop
        for i in range(65, 70):
            self.face_skeleton.append([i, i + 1])
        self.face_skeleton.append([70, 65])  # Close Right Eye loop
        # Mouth (Outer & Inner)
        for i in range(71, 82):
            self.face_skeleton.append([i, i + 1])
        self.face_skeleton.append([82, 71])  # Close Outer
        for i in range(83, 90):
            self.face_skeleton.append([i, i + 1])
        self.face_skeleton.append([90, 83])  # Close Inner

        # ---------------------------------------------------------
        # 4. HANDS (42 points): Index 91-132
        # Based on Image:
        #   Right Hand (Green box right) labels 92-112 -> Code 91-111
        #   Left Hand (Green box middle) labels 113-133 -> Code 112-132
        # ---------------------------------------------------------
        self.hand_skeleton = []

        # Helper to generate finger connections
        # wrist_idx: Index of wrist point
        # start_idx: Index of the first point of the hand block (excluding wrist logic if separate, but here part of block)
        def add_hand_edges(start_idx, wrist_conn_idx):
            # Thumb: 0->1->2->3->4
            self.hand_skeleton.append(
                [wrist_conn_idx, start_idx]
            )  # Wrist to Thumb base
            for i in range(start_idx, start_idx + 3):
                self.hand_skeleton.append([i, i + 1])

            # Fingers: Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
            # Each finger connects to wrist (0)
            for i in [5, 9, 13, 17]:
                self.hand_skeleton.append(
                    [wrist_conn_idx, start_idx + i]
                )  # Wrist to Finger Base
                for j in range(0, 3):
                    self.hand_skeleton.append(
                        [start_idx + i + j, start_idx + i + j + 1]
                    )

        # RIGHT HAND (Image 92-112 -> Code 91-111)
        # Wrist is the first point: 91
        # Connects to Body Right Wrist: Index 10 (Image 11, wait, Image 10 is Right Wrist)
        add_hand_edges(92, 91)
        self.body_hand_connections = [[10, 91]]  # Body R-Wrist to Hand R-Wrist

        # LEFT HAND (Image 113-133 -> Code 112-132)
        # Wrist is the first point: 112
        # Connects to Body Left Wrist: Index 9 (Image 10? No, Image 9 is Left Wrist)
        add_hand_edges(113, 112)
        self.body_hand_connections.append([9, 112])  # Body L-Wrist to Hand L-Wrist

    def get_full_skeleton(self):
        """Returns the complete list of [start, end] indices for visualization."""
        return (
            self.body_skeleton
            + self.feet_skeleton
            + self.face_skeleton
            + self.hand_skeleton
            + self.body_hand_connections
        )

    def get_palette(self):
        """Returns specific colors (BGR or RGB) for different parts."""
        return {
            'body': (255, 0, 0),  # Blue
            'feet': (0, 255, 0),  # Green
            'face': (255, 255, 255),  # White
            'hand': (0, 0, 255),  # Red
        }


# 使用示例
if __name__ == "__main__":
    cfg = WholeBodyConfig()
    full_skeleton = cfg.get_full_skeleton()
    print(f"Total connections defined: {len(full_skeleton)}")
    # print(full_skeleton) # Uncomment to see raw list
