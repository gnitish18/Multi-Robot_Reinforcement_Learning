def pong_ai(paddle_frect, ball_frect, table_size):
        if np.random.random() < 1:#0.5:
            if np.random.random() < 0.5:
                return "down"
            else:
                return "up"
        else:
            if paddle_frect.pos[1] + paddle_frect.size[1]/2 < ball_frect.pos[1] + ball_frect.size[1]/2:
                return "down"
            else:
                return  "up"