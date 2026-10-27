import numpy as np, cv2
def nlm_multi(frames, h, hc, tw=5):
    n=len(frames); r=tw//2; out=[]
    bgr=[cv2.cvtColor(f,cv2.COLOR_RGB2BGR) for f in frames]
    pad=[bgr[0]]*r+bgr+[bgr[-1]]*r
    for t in range(n):
        d=cv2.fastNlMeansDenoisingColoredMulti(pad[t:t+tw],r,tw,None,h,hc,5,15)
        out.append(cv2.cvtColor(d,cv2.COLOR_BGR2RGB))
    return np.array(out)
def med3(frames):
    return np.array([np.median(frames[max(0,t-1):t+2],0).astype(np.uint8) if 0<t<len(frames)-1 else frames[t] for t in range(len(frames))])
