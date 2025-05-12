if CREATE_ANIMATION:
    # Configuration
    OUTPUT_DIR            = 'pov_frames'             # where .pov and .png files go
    CYLINDER_RADIUS       = 1.125e-3                    # beam thickness
    IMAGE_WIDTH           = 800
    IMAGE_HEIGHT          = 600
    POV_BINARY            = r"C:\Program Files\POV-Ray\v3.7\bin\pvengine64.exe"                 # command for POV‑Ray
    FRAMERATE             = 30                       # fps for output video
    VIDEO_NAME            = 'beam_animation.mp4'

    # recorded_history["position"] has shape (n_frames, 3, n_nodes)
    raw_positions = recorded_history["position"]

    # Rearrange to (n_frames, n_nodes, 3)
    data = np.array(raw_positions).transpose(0, 2, 1)
    n_frames, n_nodes, _ = data.shape

    # Prepare output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for frame_idx in range(0, n_frames, 6):
        # frame_nodes is now an (n_nodes × 3) array of [x, y, z] rows
        frame_nodes = data[frame_idx]
        pov_file = os.path.join(OUTPUT_DIR, f'frame_{frame_idx:04d}.pov')
        png_file = os.path.join(OUTPUT_DIR, f'frame_{frame_idx:04d}.png')

        # Write POV‑Ray scene
        with open(pov_file, 'w') as pf:
            # White background
            pf.write('background { color rgb <1,1,1> }\n\n')
            # Camera
            pf.write('camera {\n')
            pf.write('  location <0.5, 0.5, -0.5>\n')
            pf.write('  look_at  <0, 0, 0>\n')
            pf.write('  angle 35\n')
            pf.write('}\n\n')

            # Light
            pf.write('light_source { <5, 5, -5> color rgb <1,1,1> }\n\n')

            # Beam segments
            for i in range(n_nodes - 1):
                x1, y1, z1 = frame_nodes[i]
                x2, y2, z2 = frame_nodes[i+1]
                pf.write(
                    f'cylinder {{ <{x1:.6f}, {y1:.6f}, {z1:.6f}>, '
                    f'<{x2:.6f}, {y2:.6f}, {z2:.6f}>, {CYLINDER_RADIUS:.6f} '
                    f'pigment {{ color rgb <0.8, 0.2, 0.2> }} }}\n'
                )

        # Render with POV‑Ray
        subprocess.run([
            POV_BINARY,
            '/EXIT',               # → quit when done
            f'+I{pov_file}',
            f'+O{png_file}',
            f'+W{IMAGE_WIDTH}', f'+H{IMAGE_HEIGHT}',
            '+Q9',    # quality
            '+R4'     # antialias
        ])

    # Optional: stitch into a video
    # grab all your PNGs (sorted!)
    frames = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'frame_*.png')))

    # build a clip at your target framerate
    clip = ImageSequenceClip(frames, fps=FRAMERATE)

    # write out an MP4 (uses libx264 under the hood)
    clip.write_videofile(VIDEO_NAME, codec='libx264')