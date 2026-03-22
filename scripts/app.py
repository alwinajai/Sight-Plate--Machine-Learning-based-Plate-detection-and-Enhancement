"""
app.py  —  IPDS · Indian Plate Detection System
Run: python scripts/app.py
"""

import os, sys, cv2, time, threading, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QCheckBox, QProgressBar,
    QScrollArea, QSizePolicy, QSpacerItem, QAbstractScrollArea
)
from PySide6.QtCore  import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui   import QColor, QPalette, QDragEnterEvent, QDropEvent, QCursor, QDesktopServices

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = r"E:\intern\project\Plate detection"
MODEL_PATH  = os.path.join(BASE, r"models\yolo\lp_detector_v33\weights\best.pt")
ESRGAN_PATH = os.path.join(BASE, r"models\realesrgan\realesr-general-x4v3.pth")
ANGLE_MDL   = os.path.join(BASE, r"models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\pretrained_models\angle_model.hdf5")
LENGTH_MDL  = os.path.join(BASE, r"models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\pretrained_models\length_model.hdf5")
SIDEKICK    = os.path.join(BASE, r"models\Blind-Motion-Deblurring-for-Legible-License-Plates-using-Deep-Learning\sidekick")
OUT_DETECT  = os.path.join(BASE, r"output\detected")
OUT_ENHANCE = os.path.join(BASE, r"output\enhanced")
sys.path.insert(0, SIDEKICK)
os.makedirs(OUT_DETECT, exist_ok=True)
os.makedirs(OUT_ENHANCE, exist_ok=True)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}

# ── Colour tokens ─────────────────────────────────────────────────────────────
BG       = "#0d1117"
SURF1    = "#161b22"
SURF2    = "#1c2230"
SURF3    = "#21283a"
B1       = "#21303f"
B2       = "#2d4a6a"
ACCENT   = "#238be6"
ACCENT2  = "#58a6ff"
ACCENTDM = "#1a5fa8"
T1       = "#e6edf3"
T2       = "#8b949e"
T3       = "#484f58"
T4       = "#30363d"
GREEN    = "#3fb950"
GREENBG  = "#0d2119"
AMBER    = "#d29922"
AMBERBG  = "#271d04"
RED      = "#f85149"
REDBG    = "#1f0d0c"
TOPBG    = "#090d13"
SIDBG    = "#0b1018"

# ── Stylesheet ────────────────────────────────────────────────────────────────
QSS = f"""
* {{
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
    color: {T1};
    outline: none;
}}
QWidget {{ background: {BG}; }}
QMainWindow {{ background: {BG}; }}
QLabel {{ border: none; }}
QPushButton {{ border: none; }}

#topbar {{
    background: {TOPBG};
    border-bottom: 1px solid {B1};
    min-height: 52px; max-height: 52px;
}}
#app_name  {{ font-size:14px; font-weight:700; color:{T1}; letter-spacing:1px; }}
#app_desc  {{ font-size:11px; color:{T3}; }}
#top_sep   {{ background:{B1}; min-width:1px; max-width:1px; min-height:20px; max-height:20px; }}
#tech_pill {{
    background:{SURF1}; color:{T3};
    border:1px solid {B1}; border-radius:4px;
    padding:2px 8px; font-size:10px; font-weight:500;
}}
#sidebar {{
    background:{SIDBG};
    border-right:1px solid {B1};
    min-width:272px; max-width:272px;
}}
#sec_lbl {{
    font-size:10px; font-weight:600;
    color:{T3}; letter-spacing:2px;
}}
#dropzone {{
    background:{SURF1};
    border:1.5px dashed {B2};
    border-radius:8px; min-height:106px;
}}
#dropzone_hot {{
    background:#0d1f35;
    border:1.5px dashed {ACCENT2};
    border-radius:8px; min-height:106px;
}}
#dz_icon  {{ font-size:24px; color:{B2}; }}
#dz_line1 {{ font-size:12px; font-weight:600; color:{T2}; }}
#dz_line2 {{ font-size:10px; color:{T3}; letter-spacing:1px; }}
#filebox {{
    background:{SURF1};
    border:1px solid {B1};
    border-radius:8px;
}}
#filebox_lit {{
    background:{SURF2};
    border:1px solid {B2};
    border-radius:8px;
}}
#tag_img {{ background:#12294a; color:{ACCENT2}; border-radius:4px; padding:2px 7px; font-size:10px; font-weight:700; }}
#tag_vid {{ background:#2a1060; color:#c4b5fd;   border-radius:4px; padding:2px 7px; font-size:10px; font-weight:700; }}
#tag_ext {{ background:{SURF3};  color:{T2};     border-radius:4px; padding:2px 7px; font-size:10px; font-weight:600; border:1px solid {B1}; }}
#fn_name {{ font-size:12px; font-weight:600; color:{T1}; }}
#fn_size {{ font-size:10px; color:{T2}; }}
#optbox  {{
    background:{SURF1};
    border:1px solid {B1};
    border-radius:8px;
}}
QCheckBox {{ font-size:12px; color:{T2}; spacing:8px; }}
QCheckBox:hover {{ color:{T1}; }}
QCheckBox::indicator {{
    width:15px; height:15px;
    border:1.5px solid {B2};
    border-radius:3px; background:{BG};
}}
QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT2}; }}
QCheckBox:disabled {{ color:{T4}; }}
QCheckBox::indicator:disabled {{ border-color:{T4}; background:{SURF1}; }}
#btn_run {{
    background:{ACCENT}; color:#ffffff;
    border-radius:6px; font-size:12px;
    font-weight:700; letter-spacing:1px;
    min-height:40px; padding:0;
}}
#btn_run:hover   {{ background:{ACCENT2}; }}
#btn_run:pressed {{ background:{ACCENTDM}; }}
#btn_run:disabled {{ background:{SURF2}; color:{T3}; }}
#btn_ghost {{
    background:transparent; color:{T2};
    border:1px solid {B1}; border-radius:6px;
    font-size:11px; min-height:32px; padding:0 12px;
}}
#btn_ghost:hover {{ background:{SURF3}; color:{T1}; border-color:{B2}; }}
#btn_danger {{
    background:transparent; color:{T3};
    border:1px solid {B1}; border-radius:6px;
    font-size:11px; min-height:32px; padding:0 12px;
}}
#btn_danger:hover {{ background:{REDBG}; color:{RED}; border-color:#5c2020; }}
#stat_card {{
    background:{SURF1}; border:1px solid {B1};
    border-radius:8px; min-height:72px;
}}
#stat_val {{ font-size:26px; font-weight:700; color:{T1}; }}
#stat_key {{ font-size:10px; color:{T3}; letter-spacing:1px; }}
#panel_head {{
    background:{SURF1};
    border:1px solid {B1};
    border-bottom: 1px solid {B1};
    border-top-left-radius:8px; border-top-right-radius:8px;
    min-height:36px; max-height:36px;
}}
#panel_body {{
    background:{SURF1}; border:1px solid {B1};
    border-top:none;
    border-bottom-left-radius:8px; border-bottom-right-radius:8px;
}}
#panel_title {{ font-size:11px; font-weight:600; color:{T2}; letter-spacing:1px; }}
QProgressBar {{
    background:{SURF3}; border:none;
    border-radius:2px; max-height:4px;
}}
QProgressBar::chunk {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {ACCENTDM}, stop:1 {ACCENT2});
    border-radius:2px;
}}
#prog_msg {{ font-size:11px; color:{T2}; }}
#prog_pct {{ font-size:11px; font-weight:700; color:{ACCENT2}; }}
#res_row {{
    background:{SURF1}; border:1px solid {B1};
    border-radius:6px; min-height:50px;
}}
#res_row:hover {{ border-color:{B2}; background:{SURF2}; }}
#badge_hi {{ background:{GREENBG}; color:{GREEN}; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }}
#badge_md {{ background:{AMBERBG}; color:{AMBER}; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }}
#badge_lo {{ background:{REDBG};   color:{RED};   border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }}
#res_name {{ font-size:12px; font-weight:600; color:{T1}; }}
#res_sub  {{ font-size:10px; color:{T2}; }}
QScrollBar:vertical {{ background:{BG}; width:4px; border:none; margin:0; }}
QScrollBar::handle:vertical {{ background:{B2}; border-radius:2px; min-height:20px; }}
QScrollBar::handle:vertical:hover {{ background:{T3}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar::add-page:vertical,  QScrollBar::sub-page:vertical {{ background:none; }}
#footer {{
    background:{TOPBG}; border-top:1px solid {B1};
    min-height:32px; max-height:32px;
}}
#ft_text {{ font-size:10px; color:{T4}; }}
#ft_name {{ font-size:10px; font-weight:600; color:{T3}; }}
#ft_link {{ background:transparent; color:{ACCENT}; border:none; font-size:10px; font-weight:600; padding:0; min-height:0; }}
#ft_link:hover {{ color:{ACCENT2}; }}
#ft_sep  {{ font-size:10px; color:{T4}; padding:0 8px; }}
#chip {{ border-radius:10px; padding:3px 10px; font-size:10px; font-weight:600; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER
# ─────────────────────────────────────────────────────────────────────────────
class Worker(QThread):
    sig_prog  = Signal(int, str)
    sig_log   = Signal(str, str)
    sig_plate = Signal(str, float, bool)
    sig_done  = Signal(bool, str)

    def __init__(self, path, enhance, snaps):
        super().__init__()
        self.path=path; self.enhance=enhance; self.snaps=snaps; self._stop=False
        self.yolo=self.ang=self.lng=self.sr=None

    def stop(self): self._stop=True

    def _fft(self,g):
        f=np.fft.fftshift(np.fft.fft2(g))
        m=20*np.log(np.abs(f)+1e-8)
        m=cv2.normalize(m,None,0,255,cv2.NORM_MINMAX)
        return cv2.resize(m.astype(np.uint8),(224,224))

    def _deblur(self,img):
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        inp=np.expand_dims(np.expand_dims(self._fft(g).astype(np.float32)/255.,-1),0)
        angle=int(np.argmax(self.ang.predict(inp,verbose=0)[0]))%180
        length=max(1,min(int(np.argmax(self.lng.predict(inp,verbose=0)[0]))+1,30))
        if length<=1: return img
        k=np.zeros((length,length),np.float32); k[length//2,:]=1./length
        M=cv2.getRotationMatrix2D((length//2,length//2),angle,1.0)
        k=cv2.warpAffine(k,M,(length,length)); k/=k.sum()+1e-8
        res=np.zeros_like(img,np.float32)
        for c in range(3):
            ch=img[:,:,c].astype(np.float32)/255.
            kp=np.zeros_like(ch); kp[:length,:length]=k
            w=np.conj(np.fft.fft2(kp))/(np.abs(np.fft.fft2(kp))**2+0.01)
            res[:,:,c]=(np.clip(np.fft.ifft2(np.fft.fft2(ch)*w).real,0,1)*255).astype(np.uint8)
        return res.astype(np.uint8)

    def _deskew(self,img):
        e=cv2.Canny(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),50,150,apertureSize=3)
        L=cv2.HoughLinesP(e,1,np.pi/180,20,minLineLength=img.shape[1]//5,maxLineGap=10)
        if L is None: return img
        a=[np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0])) for l in L if l[0][2]-l[0][0]!=0]
        a=[x for x in a if -45<x<45]
        if not a or abs(np.median(a))<1.5: return img
        ang=float(np.median(a)); h,w=img.shape[:2]
        M=cv2.getRotationMatrix2D((w//2,h//2),ang,1.0)
        ca,sa=abs(M[0,0]),abs(M[0,1])
        nw,nh=int(h*sa+w*ca),int(h*ca+w*sa)
        M[0,2]+=(nw-w)/2; M[1,2]+=(nh-h)/2
        return cv2.warpAffine(img,M,(nw,nh),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    def _upscale(self,img):
        h,w=img.shape[:2]
        if w>120 and h>40:
            up=cv2.resize(img,(w*4,h*4),interpolation=cv2.INTER_CUBIC)
            return cv2.addWeighted(up,1.4,cv2.GaussianBlur(up,(0,0),1.5),-0.4,0)
        pre=cv2.resize(img,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
        e,_=self.sr.enhance(pre,outscale=4)
        return cv2.addWeighted(e,1.4,cv2.GaussianBlur(e,(0,0),2.0),-0.4,0)

    def _binary(self,img):
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        g=cv2.createCLAHE(2.5,(4,4)).apply(g)
        g=cv2.fastNlMeansDenoising(g,h=5)
        g=cv2.GaussianBlur(g,(3,3),0)
        _,ot=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ad=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,10)
        wr=np.sum(ot==255)/ot.size
        b=ot if 0.45<wr<0.85 else ad
        if np.sum(b==255)/b.size<0.5: b=cv2.bitwise_not(b)
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k)
        nb,labels,stats,_=cv2.connectedComponentsWithStats(cv2.bitwise_not(b),8)
        cl=np.full_like(b,255)
        for i in range(1,nb):
            if stats[i,cv2.CC_STAT_AREA]>=20: cl[labels==i]=0
        return cv2.cvtColor(cl,cv2.COLOR_GRAY2BGR)

    def _do_enhance(self,crop_path,base,idx):
        img=cv2.imread(crop_path)
        if img is None: return False
        img=self._deblur(img); img=self._deskew(img); img=self._upscale(img)
        bn=self._binary(img.copy())
        cv2.imwrite(os.path.join(OUT_ENHANCE,f"{base}_p{idx}_color.jpg"),img)
        cv2.imwrite(os.path.join(OUT_ENHANCE,f"{base}_p{idx}_binary.jpg"),bn)
        return True

    def _image(self):
        self.sig_log.emit("Reading image","info")
        img=cv2.imread(self.path)
        base=os.path.splitext(os.path.basename(self.path))[0]
        self.sig_prog.emit(10,"Running detection...")
        res=self.yolo.predict(source=self.path,conf=0.25,device=0,verbose=False)
        boxes=res[0].boxes
        if not len(boxes):
            self.sig_log.emit("No plates found","warn")
            self.sig_done.emit(False,"No plates detected.")
            return
        total=len(boxes)
        self.sig_log.emit(f"Found {total} plate(s)","ok")
        for i,box in enumerate(boxes):
            if self._stop: return
            x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
            conf=float(box.conf[0])
            h,w=img.shape[:2]
            crop=img[max(0,y1-5):min(h,y2+5),max(0,x1-5):min(w,x2+5)]
            cp=os.path.join(OUT_DETECT,f"{base}_plate{i+1}.jpg")
            cv2.imwrite(cp,crop)
            cv2.rectangle(img,(x1,y1),(x2,y2),(35,139,230),2)
            cv2.putText(img,f"{conf:.2f}",(x1,max(y1-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(35,139,230),2)
            pct=15+int((i+1)/total*(55 if self.enhance else 75))
            self.sig_prog.emit(pct,f"Plate {i+1}/{total}...")
            enhanced=False
            if self.enhance and self.sr:
                self.sig_log.emit(f"Enhancing plate {i+1}...","info")
                enhanced=self._do_enhance(cp,base,i+1)
            self.sig_plate.emit(cp,conf,enhanced)
            self.sig_log.emit(f"Plate {i+1}  conf={conf:.2f}  {'enhanced' if enhanced else 'saved'}","ok")
        cv2.imwrite(os.path.join(OUT_DETECT,f"{base}_annotated.jpg"),img)
        self.sig_log.emit("Annotated image saved","ok")
        self.sig_prog.emit(100,"Complete")
        self.sig_done.emit(True,f"{total} plate(s)"+(" enhanced" if self.enhance else " detected"))

    def _video(self):
        self.sig_log.emit("Opening video...","info")
        cap=cv2.VideoCapture(self.path)
        base=os.path.splitext(os.path.basename(self.path))[0]
        tf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS) or 25
        W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out=os.path.join(OUT_DETECT,f"{base}_annotated.mp4")
        wri=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
        sd=os.path.join(OUT_DETECT,f"{base}_snapshots")
        if self.snaps: os.makedirs(sd,exist_ok=True)
        self.sig_log.emit(f"{W}×{H}  {fps:.0f}fps  {tf} frames","info")
        fi=0; pc=0; last=[]; enh_every=15
        while True:
            ret,frame=cap.read()
            if not ret or self._stop: break
            if fi%2==0:
                r=self.yolo.predict(source=frame,conf=0.25,device=0,verbose=False,stream=False)
                last=r[0].boxes
            for box in last:
                x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
                conf=float(box.conf[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(35,139,230),2)
                cv2.putText(frame,f"LP {conf:.2f}",(x1,max(y1-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(35,139,230),2)
                # Save crop on every detection frame
                if fi%2==0:
                    pc+=1
                    cr=frame[max(0,y1-5):min(H,y2+5),max(0,x1-5):min(W,x2+5)]
                    cp=os.path.join(OUT_DETECT,f"{base}_f{fi:05d}_p{pc}.jpg")
                    cv2.imwrite(cp,cr)
                    # Also save to snapshots folder if requested
                    if self.snaps:
                        cv2.imwrite(os.path.join(sd,f"f{fi:05d}_p{pc}.jpg"),cr)
                    # Enhance every Nth plate to avoid slowdown on long videos
                    enhanced=False
                    if self.enhance and self.sr and pc%enh_every==0:
                        enhanced=self._do_enhance(cp,base,pc)
                    # Emit to update stats and results panel
                    self.sig_plate.emit(cp,conf,enhanced)
            cv2.putText(frame,f"{fi}/{tf}",(8,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(40,60,90),1)
            wri.write(frame)
            if fi%25==0:
                self.sig_prog.emit(int(fi/max(tf,1)*95),f"Frame {fi}/{tf}  —  {pc} plates found")
            fi+=1
        cap.release(); wri.release()
        # After video is done, run enhancement on all saved crops if enhance is on
        if self.enhance and self.sr and pc>0:
            self.sig_log.emit(f"Enhancing all {pc} saved plate crops...","info")
            crops=[os.path.join(OUT_DETECT,f) for f in os.listdir(OUT_DETECT)
                   if f.startswith(base+"_f") and f.endswith(".jpg")]
            for idx,cp in enumerate(crops):
                if self._stop: break
                self._do_enhance(cp,base,idx+1)
                if idx%50==0:
                    pct=int(idx/max(len(crops),1)*100)
                    self.sig_log.emit(f"Enhanced {idx+1}/{len(crops)} crops...","info")
            self.sig_log.emit(f"All {len(crops)} crops enhanced","ok")
        self.sig_log.emit("Annotated video saved","ok")
        if self.snaps: self.sig_log.emit(f"{pc} snapshots saved","ok")
        self.sig_prog.emit(100,"Complete")
        self.sig_done.emit(True,f"{pc} plates in {fi} frames"+(" (enhanced)" if self.enhance else ""))

    def run(self):
        try:
            ext=os.path.splitext(self.path)[1].lower()
            if ext in IMAGE_EXTS: self._image()
            elif ext in VIDEO_EXTS: self._video()
        except Exception as e:
            self.sig_log.emit(f"Error: {e}","err")
            self.sig_done.emit(False,str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  DROP ZONE
# ─────────────────────────────────────────────────────────────────────────────
class DropZone(QWidget):
    dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("dropzone")
        self.setAcceptDrops(True)
        self.setFixedHeight(106)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0,0,0,0)
        vl.setSpacing(5)
        vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ico = QLabel("⬆"); ico.setObjectName("dz_icon")
        ico.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ico.setStyleSheet(f"font-size:24px; color:{B2};")
        l1 = QLabel("Drop file here  or  click to browse"); l1.setObjectName("dz_line1")
        l1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l1.setStyleSheet(f"font-size:12px; font-weight:600; color:{T2};")
        l2 = QLabel("JPG · PNG · BMP · MP4 · AVI · MOV · MKV"); l2.setObjectName("dz_line2")
        l2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l2.setStyleSheet(f"font-size:10px; color:{T3}; letter-spacing:1px;")
        for w in [ico, l1, l2]: vl.addWidget(w)

    def _emit(self, p):
        if p: self.dropped.emit(p)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setObjectName("dropzone_hot")
            self.setStyleSheet(f"background:#0d1f35; border:1.5px dashed {ACCENT2}; border-radius:8px;")

    def dragLeaveEvent(self, e):
        self.setObjectName("dropzone"); self.setStyleSheet("")

    def dropEvent(self, e: QDropEvent):
        self.dragLeaveEvent(e)
        u = e.mimeData().urls()
        if u: self._emit(u[0].toLocalFile())

    def mousePressEvent(self, e):
        p, _ = QFileDialog.getOpenFileName(self, "Select File", "",
            "Supported (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.wmv);;"
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;"
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v)")
        self._emit(p)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sec_label(text):
    l = QLabel(text.upper()); l.setObjectName("sec_lbl")
    l.setStyleSheet(f"font-size:10px; font-weight:600; color:{T3}; letter-spacing:2px;")
    l.setFixedHeight(18)
    return l


def panel(title, body_widget, body_height=None):
    """Returns a complete panel: titled header bar + content body as one widget."""
    wrap = QWidget()
    vl   = QVBoxLayout(wrap); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
    # Header
    hd = QWidget(); hd.setObjectName("panel_head")
    hd.setFixedHeight(36)
    hl = QHBoxLayout(hd); hl.setContentsMargins(14,0,14,0)
    tl = QLabel(title.upper()); tl.setObjectName("panel_title")
    tl.setStyleSheet(f"font-size:11px; font-weight:600; color:{T2}; letter-spacing:1px;")
    hl.addWidget(tl); hl.addStretch()
    vl.addWidget(hd)
    # Body
    bd = QWidget(); bd.setObjectName("panel_body")
    bl = QVBoxLayout(bd); bl.setContentsMargins(0,0,0,0); bl.setSpacing(0)
    bl.addWidget(body_widget)
    if body_height:
        bd.setFixedHeight(body_height)
    vl.addWidget(bd)
    return wrap, hl, bl   # return header layout too so caller can add buttons


def stat_card(key, init="0"):
    w = QWidget(); w.setObjectName("stat_card")
    w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    w.setFixedHeight(72)
    vl = QVBoxLayout(w); vl.setContentsMargins(16,12,16,12); vl.setSpacing(2)
    vl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    val = QLabel(init); val.setObjectName("stat_val")
    val.setAlignment(Qt.AlignmentFlag.AlignCenter)
    val.setStyleSheet(f"font-size:26px; font-weight:700; color:{T1};")
    lbl = QLabel(key.upper()); lbl.setObjectName("stat_key")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet(f"font-size:9px; color:{T3}; letter-spacing:1px;")
    vl.addWidget(val); vl.addWidget(lbl)
    return w, val


# ─────────────────────────────────────────────────────────────────────────────
#  LOG WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class LogWidget(QWidget):
    C = {"info":T2, "ok":GREEN, "warn":AMBER, "err":RED}
    P = {"info":"·", "ok":"✓", "warn":"▲", "err":"✕"}

    def __init__(self):
        super().__init__()
        vl = QVBoxLayout(self); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
        # header bar
        hd = QWidget(); hd.setFixedHeight(36)
        hd.setStyleSheet(
            f"background:{SURF1};"
            f"border-top:1px solid {B1};"
            f"border-left:1px solid {B1};"
            f"border-right:1px solid {B1};"
            f"border-bottom:1px solid {B1};"
            f"border-top-left-radius:8px;"
            f"border-top-right-radius:8px;"
        )
        hl = QHBoxLayout(hd); hl.setContentsMargins(14,0,14,0)
        tl = QLabel("SYSTEM LOG")
        tl.setStyleSheet(f"font-size:11px; font-weight:600; color:{T2}; letter-spacing:1px; border:none;")
        cb = QPushButton("Clear"); cb.setFixedHeight(22)
        cb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        cb.setStyleSheet(f"background:transparent; color:{T3}; border:none; font-size:10px; padding:0 6px;")
        cb.clicked.connect(self.clear)
        hl.addWidget(tl); hl.addStretch(); hl.addWidget(cb)
        vl.addWidget(hd)
        # scroll body
        bd = QWidget()
        bd.setStyleSheet(
            f"background:{SURF1};"
            f"border-left:1px solid {B1};"
            f"border-right:1px solid {B1};"
            f"border-bottom:1px solid {B1};"
            f"border-top:none;"
            f"border-bottom-left-radius:8px;"
            f"border-bottom-right-radius:8px;"
        )
        bd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        bl = QVBoxLayout(bd); bl.setContentsMargins(0,0,0,0); bl.setSpacing(0)
        self._sc = QScrollArea(); self._sc.setWidgetResizable(True)
        self._sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sc.setStyleSheet("background:transparent; border:none;")
        self._iw = QWidget(); self._iw.setStyleSheet("background:transparent;")
        self._il = QVBoxLayout(self._iw); self._il.setContentsMargins(12,8,12,8); self._il.setSpacing(2)
        self._il.addStretch()
        self._sc.setWidget(self._iw)
        bl.addWidget(self._sc)
        vl.addWidget(bd)

    def add(self, msg, level="info"):
        c=self.C.get(level,T2); p=self.P.get(level,"·"); ts=time.strftime("%H:%M:%S")
        rw=QWidget(); rw.setStyleSheet("background:transparent;")
        rl=QHBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(8)
        tsl=QLabel(ts); tsl.setFixedWidth(52)
        tsl.setStyleSheet(f"color:{T4}; font-size:10px; font-family:'Courier New';")
        pl=QLabel(p); pl.setFixedWidth(10); pl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.setStyleSheet(f"color:{c}; font-size:11px;")
        ml=QLabel(msg); ml.setWordWrap(True)
        ml.setStyleSheet(f"color:{c}; font-size:11px;")
        rl.addWidget(tsl); rl.addWidget(pl); rl.addWidget(ml,1)
        self._il.insertWidget(self._il.count()-1, rw)
        QTimer.singleShot(30, lambda: self._sc.verticalScrollBar().setValue(
            self._sc.verticalScrollBar().maximum()))

    def clear(self):
        while self._il.count()>1:
            it=self._il.takeAt(0)
            if it.widget(): it.widget().deleteLater()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IPDS  —  Indian Plate Detection System")
        self.setMinimumSize(1000, 700)
        self.resize(1180, 820)
        self.setStyleSheet(QSS)
        self._file=None; self._video=False
        self._worker=None; self._ready=False
        self._nd=0; self._ne=0; self._ns=0
        self._yolo=self._ang=self._lng=self._sr=None
        self._log=LogWidget()
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._rph=None
        self._build()
        QTimer.singleShot(300, self._load_models)

    # ── Full UI build ─────────────────────────────────────────────────────────
    def _build(self):
        root=QWidget(); self.setCentralWidget(root)
        vl=QVBoxLayout(root); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)
        vl.addWidget(self._mk_topbar())
        body=QWidget(); bl=QHBoxLayout(body)
        bl.setContentsMargins(0,0,0,0); bl.setSpacing(0)
        bl.addWidget(self._mk_sidebar())
        bl.addWidget(self._mk_main(),1)
        vl.addWidget(body,1)
        vl.addWidget(self._mk_footer())

    # ── Topbar ────────────────────────────────────────────────────────────────
    def _mk_topbar(self):
        tb=QWidget(); tb.setObjectName("topbar"); tb.setFixedHeight(52)
        hl=QHBoxLayout(tb); hl.setContentsMargins(20,0,20,0); hl.setSpacing(12)
        # title block
        tv=QVBoxLayout(); tv.setSpacing(1); tv.setContentsMargins(0,0,0,0)
        nm_row=QHBoxLayout(); nm_row.setSpacing(7); nm_row.setContentsMargins(0,0,0,0)
        dot=QLabel("●"); dot.setFixedWidth(12)
        dot.setStyleSheet(f"font-size:8px; color:{ACCENT2}; border:none;")
        nm=QLabel("IPDS")
        nm.setStyleSheet(f"font-size:14px; font-weight:700; color:{T1}; letter-spacing:1px; border:none;")
        nm_row.addWidget(dot); nm_row.addWidget(nm)
        ds=QLabel("Indian Plate Detection System")
        ds.setStyleSheet(f"font-size:11px; color:{T3}; border:none;")
        tv.addLayout(nm_row); tv.addWidget(ds)
        hl.addLayout(tv)
        hl.addSpacing(20)
        for t in ["YOLOv8s", "ESRGAN v3", "Blind Deblur", "CUDA"]:
            b=QLabel(t)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            b.setStyleSheet(
                f"background:{SURF2}; color:{T2}; border:none;"
                f"border-radius:4px; padding:3px 10px; font-size:10px; font-weight:500;"
            )
            hl.addWidget(b)
        hl.addStretch()
        self._chip=QLabel("● LOADING MODELS"); self._chip.setObjectName("chip")
        self._chip.setStyleSheet(
            f"background:{SURF2}; color:{T2}; border:none;"
            f"border-radius:10px; padding:3px 12px; font-size:10px; font-weight:600;"
        )
        hl.addWidget(self._chip)
        return tb

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _mk_sidebar(self):
        sb=QWidget(); sb.setObjectName("sidebar"); sb.setFixedWidth(272)
        vl=QVBoxLayout(sb); vl.setContentsMargins(14,16,14,16); vl.setSpacing(0)

        # ── Upload ──
        vl.addWidget(sec_label("Upload"))
        vl.addSpacing(6)
        self._dz=DropZone(); self._dz.dropped.connect(self._on_file)
        vl.addWidget(self._dz)
        vl.addSpacing(14)

        # ── File Info ──
        vl.addWidget(sec_label("File Info"))
        vl.addSpacing(6)
        self._fbox=QWidget(); self._fbox.setObjectName("filebox")
        self._fbox.setStyleSheet(f"background:{SURF1}; border:1px solid {B1}; border-style:solid; border-radius:8px;")
        self._fbox_vl=QVBoxLayout(self._fbox)
        self._fbox_vl.setContentsMargins(12,10,12,10); self._fbox_vl.setSpacing(6)
        self._fbox_ph=QLabel("No file selected")
        self._fbox_ph.setStyleSheet(f"color:{T3}; font-size:11px;")
        self._fbox_vl.addWidget(self._fbox_ph)
        vl.addWidget(self._fbox)
        vl.addSpacing(14)

        # ── Options ──
        vl.addWidget(sec_label("Options"))
        vl.addSpacing(6)
        ob=QWidget(); ob.setObjectName("optbox")
        ob.setStyleSheet(f"background:{SURF1}; border:1px solid {B1}; border-style:solid; border-radius:8px;")
        ovl=QVBoxLayout(ob); ovl.setContentsMargins(12,10,12,10); ovl.setSpacing(8)
        self._chk_enh=QCheckBox("Enhance detected plates")
        self._chk_enh.setChecked(True)
        self._chk_snap=QCheckBox("Save frame snapshots")
        self._chk_snap.setChecked(False); self._chk_snap.setVisible(False)
        ovl.addWidget(self._chk_enh); ovl.addWidget(self._chk_snap)
        vl.addWidget(ob)
        vl.addSpacing(14)

        # ── Analyze button ──
        self._btn_run=QPushButton("ANALYZE")
        self._btn_run.setObjectName("btn_run")
        self._btn_run.setFixedHeight(40)
        self._btn_run.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run)
        vl.addWidget(self._btn_run)
        vl.addSpacing(8)

        # ── Action row ──
        ar=QHBoxLayout(); ar.setSpacing(8); ar.setContentsMargins(0,0,0,0)
        self._btn_clr=QPushButton("Clear"); self._btn_clr.setObjectName("btn_danger")
        self._btn_clr.setFixedHeight(32); self._btn_clr.setEnabled(False)
        self._btn_clr.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_clr.clicked.connect(self._clear)
        self._btn_out=QPushButton("Open Output"); self._btn_out.setObjectName("btn_ghost")
        self._btn_out.setFixedHeight(32)
        self._btn_out.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_out.clicked.connect(self._open_out)
        ar.addWidget(self._btn_clr,1); ar.addWidget(self._btn_out,1)
        vl.addLayout(ar)

        vl.addStretch()
        return sb

    # ── Main ──────────────────────────────────────────────────────────────────
    def _mk_main(self):
        mw=QWidget(); mw.setObjectName("main_area")
        vl=QVBoxLayout(mw); vl.setContentsMargins(14,16,14,14); vl.setSpacing(0)

        # ── Stat cards ──
        vl.addWidget(sec_label("Overview"))
        vl.addSpacing(6)
        sr=QHBoxLayout(); sr.setSpacing(10); sr.setContentsMargins(0,0,0,0)
        sc1,self._sv_det=stat_card("Plates Detected")
        sc2,self._sv_enh=stat_card("Plates Enhanced")
        sc3,self._sv_ses=stat_card("Session Files")
        for s in [sc1,sc2,sc3]: sr.addWidget(s)
        vl.addLayout(sr)
        vl.addSpacing(14)

        # ── Progress ──
        vl.addWidget(sec_label("Progress"))
        vl.addSpacing(6)
        pc=QWidget()
        pc.setStyleSheet(f"background:{SURF1}; border:1px solid {B1}; border-radius:8px; border-style:solid;")
        pvl=QVBoxLayout(pc); pvl.setContentsMargins(14,12,14,12); pvl.setSpacing(8)
        ph=QHBoxLayout(); ph.setContentsMargins(0,0,0,0)
        self._pmsg=QLabel("Waiting for input...")
        self._pmsg.setObjectName("prog_msg")
        self._pmsg.setStyleSheet(f"font-size:11px; color:{T2};")
        self._ppct=QLabel("0%")
        self._ppct.setObjectName("prog_pct")
        self._ppct.setStyleSheet(f"font-size:11px; font-weight:700; color:{ACCENT2};")
        ph.addWidget(self._pmsg); ph.addStretch(); ph.addWidget(self._ppct)
        self._pbar=QProgressBar(); self._pbar.setValue(0)
        self._pbar.setTextVisible(False); self._pbar.setFixedHeight(4)
        pvl.addLayout(ph); pvl.addWidget(self._pbar)
        vl.addWidget(pc)
        vl.addSpacing(12)

        # ── Bottom two-column ──
        bot=QHBoxLayout(); bot.setSpacing(12); bot.setContentsMargins(0,0,0,0)

        # Results
        rv=QVBoxLayout(); rv.setSpacing(0); rv.setContentsMargins(0,0,0,0)
        rv.addWidget(sec_label("Results"))
        rv.addSpacing(6)
        # results panel wrapper (header + scroll body together)
        rpw=QWidget()
        rpw.setStyleSheet(f"background:transparent;")
        rpw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rvl=QVBoxLayout(rpw); rvl.setContentsMargins(0,0,0,0); rvl.setSpacing(0)
        # header bar
        rhd=QWidget(); rhd.setFixedHeight(36)
        rhd.setStyleSheet(
            f"background:{SURF1};"
            f"border-top:1px solid {B1};"
            f"border-left:1px solid {B1};"
            f"border-right:1px solid {B1};"
            f"border-bottom:1px solid {B1};"
            f"border-top-left-radius:8px;"
            f"border-top-right-radius:8px;"
        )
        rhl=QHBoxLayout(rhd); rhl.setContentsMargins(14,0,14,0)
        rtl=QLabel("DETECTED PLATES")
        rtl.setStyleSheet(f"font-size:11px; font-weight:600; color:{T2}; letter-spacing:1px; border:none;")
        self._r_count=QLabel("0 results")
        self._r_count.setStyleSheet(f"font-size:11px; color:{T3}; border:none;")
        rhl.addWidget(rtl); rhl.addStretch(); rhl.addWidget(self._r_count)
        rvl.addWidget(rhd)
        # scroll
        rbd=QWidget()
        rbd.setStyleSheet(
            f"background:{SURF1};"
            f"border-left:1px solid {B1};"
            f"border-right:1px solid {B1};"
            f"border-bottom:1px solid {B1};"
            f"border-top:none;"
            f"border-bottom-left-radius:8px;"
            f"border-bottom-right-radius:8px;"
        )
        rbd.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rbl=QVBoxLayout(rbd); rbl.setContentsMargins(0,0,0,0); rbl.setSpacing(0)
        self._rsc=QScrollArea(); self._rsc.setWidgetResizable(True)
        self._rsc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._rsc.setStyleSheet("background:transparent; border:none;")
        self._riw=QWidget(); self._riw.setStyleSheet(f"background:{SURF1};")
        self._ril=QVBoxLayout(self._riw)
        self._ril.setContentsMargins(10,10,10,10); self._ril.setSpacing(6)
        self._rph=QLabel("No results yet")
        self._rph.setStyleSheet(f"color:{T3}; font-size:11px;")
        self._rph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ril.addWidget(self._rph); self._ril.addStretch()
        self._rsc.setWidget(self._riw)
        rbl.addWidget(self._rsc)
        rvl.addWidget(rbd)
        rv.addWidget(rpw,1)
        bot.addLayout(rv,1)

        # Log
        lv=QVBoxLayout(); lv.setSpacing(0); lv.setContentsMargins(0,0,0,0)
        lv.addWidget(sec_label("Log"))
        lv.addSpacing(6)
        lv.addWidget(self._log,1)
        bot.addLayout(lv,1)

        vl.addLayout(bot,1)
        return mw

    # ── Footer ────────────────────────────────────────────────────────────────
    def _mk_footer(self):
        ft=QWidget(); ft.setObjectName("footer"); ft.setFixedHeight(32)
        hl=QHBoxLayout(ft); hl.setContentsMargins(20,0,20,0); hl.setSpacing(0)
        lt=QLabel("IPDS  ·  Indian Plate Detection System  ·  v3.0")
        lt.setObjectName("ft_text")
        lt.setStyleSheet(f"font-size:10px; color:{T4};")
        hl.addWidget(lt); hl.addStretch()
        s1=QLabel("·"); s1.setObjectName("ft_sep")
        s1.setStyleSheet(f"font-size:10px; color:{T4}; padding:0 10px;")
        nm=QLabel("Developed by  Alwin Ajai"); nm.setObjectName("ft_name")
        nm.setStyleSheet(f"font-size:10px; font-weight:600; color:{T3};")
        s2=QLabel("·"); s2.setObjectName("ft_sep")
        s2.setStyleSheet(f"font-size:10px; color:{T4}; padding:0 10px;")
        lk=QPushButton("LinkedIn  ↗"); lk.setObjectName("ft_link")
        lk.setFixedHeight(20)
        lk.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        lk.setStyleSheet(f"background:transparent; color:{ACCENT}; border:none; "
                         f"font-size:10px; font-weight:600; padding:0;")
        lk.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://www.linkedin.com/in/alwin-ajai-817436201/")))
        hl.addWidget(s1); hl.addWidget(nm); hl.addWidget(s2); hl.addWidget(lk)
        return ft

    # ── File handling ─────────────────────────────────────────────────────────
    def _on_file(self, path):
        if not os.path.exists(path): return
        self._file=path
        self._video=os.path.splitext(path)[1].lower() in VIDEO_EXTS
        self._chk_snap.setVisible(self._video)
        self._refresh_filebox()
        self._clear_results()
        self._reset_prog()
        self._log.clear()
        self._log.add(f"Loaded: {os.path.basename(path)}","ok")
        self._btn_clr.setEnabled(True)
        self._ns+=1; self._sv_ses.setText(str(self._ns))
        if self._ready: self._btn_run.setEnabled(True)

    def _clear(self):
        self._file=None; self._video=False
        self._chk_snap.setVisible(False)
        self._refresh_filebox()
        self._clear_results(); self._reset_prog()
        self._btn_run.setEnabled(False); self._btn_clr.setEnabled(False)
        self._log.clear()

    def _refresh_filebox(self):
        # wipe existing children
        while self._fbox_vl.count():
            it=self._fbox_vl.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        if not self._file:
            ph=QLabel("No file selected")
            ph.setStyleSheet(f"color:{T3}; font-size:11px;")
            self._fbox_vl.addWidget(ph)
            self._fbox.setObjectName("filebox")
            self._fbox.setStyleSheet(f"background:{SURF1}; border:1px solid {B1}; border-style:solid; border-radius:8px;")
            return
        # active state
        self._fbox.setObjectName("filebox_lit")
        self._fbox.setStyleSheet(f"background:{SURF2}; border:1px solid {B2}; border-radius:8px;")
        fname=os.path.basename(self._file)
        sz=os.path.getsize(self._file)
        szs=f"{sz/1024/1024:.1f} MB" if sz>1024*1024 else f"{sz/1024:.1f} KB"
        ext=os.path.splitext(self._file)[1].upper()
        ftype="VIDEO" if self._video else "IMAGE"
        # badge row
        br=QWidget(); br.setStyleSheet("background:transparent;")
        brl=QHBoxLayout(br); brl.setContentsMargins(0,0,0,0); brl.setSpacing(6)
        tb=QLabel(ftype)
        tb.setStyleSheet(f"background:{'#2a1060' if self._video else '#12294a'}; "
                         f"color:{'#c4b5fd' if self._video else ACCENT2}; "
                         f"border-radius:4px; padding:2px 7px; font-size:10px; font-weight:700;")
        eb=QLabel(ext)
        eb.setStyleSheet(f"background:{SURF3}; color:{T2}; border:1px solid {B1}; "
                         f"border-radius:4px; padding:2px 7px; font-size:10px; font-weight:600;")
        brl.addWidget(tb); brl.addWidget(eb); brl.addStretch()
        nl=QLabel(fname); nl.setObjectName("fn_name")
        nl.setWordWrap(True)
        nl.setStyleSheet(f"font-size:12px; font-weight:600; color:{T1};")
        sl=QLabel(szs); sl.setObjectName("fn_size")
        sl.setStyleSheet(f"font-size:10px; color:{T2};")
        for w in [br,nl,sl]: self._fbox_vl.addWidget(w)

    # ── Analysis ──────────────────────────────────────────────────────────────
    def _run(self):
        if not self._file or not self._ready: return
        self._btn_run.setEnabled(False)
        self._clear_results(); self._reset_prog()
        self._nd=0; self._ne=0
        self._sv_det.setText("0"); self._sv_enh.setText("0")
        self._set_chip("PROCESSING",ACCENT2,f"background:#0d1f35; border:1px solid {ACCENT};")
        self._log.clear(); self._log.add("Analysis started","info")
        self._worker=Worker(self._file, self._chk_enh.isChecked(),
                            self._chk_snap.isChecked() and self._video)
        self._worker.yolo=self._yolo; self._worker.ang=self._ang
        self._worker.lng=self._lng;   self._worker.sr=self._sr
        self._worker.sig_prog.connect(self._on_prog)
        self._worker.sig_log.connect(self._log.add)
        self._worker.sig_plate.connect(self._on_plate)
        self._worker.sig_done.connect(self._on_done)
        self._worker.start()

    def _on_prog(self,pct,msg):
        self._pbar.setValue(pct); self._ppct.setText(f"{pct}%"); self._pmsg.setText(msg)

    def _on_plate(self,path,conf,enhanced):
        self._nd+=1
        if enhanced: self._ne+=1
        self._sv_det.setText(str(self._nd)); self._sv_enh.setText(str(self._ne))
        self._r_count.setText(f"{self._nd} result{'s' if self._nd!=1 else ''}")
        # remove placeholder
        if self._rph and self._rph.isVisible(): self._rph.setVisible(False)
        # result row
        rw=QWidget(); rw.setObjectName("res_row")
        rw.setStyleSheet(f"background:{SURF1}; border:1px solid {B1}; border-radius:6px;")
        rw.setFixedHeight(52)
        rl=QHBoxLayout(rw); rl.setContentsMargins(12,0,12,0); rl.setSpacing(10)
        # conf badge
        if conf>=0.80:   bg,tc,obj=GREENBG,GREEN,"badge_hi"
        elif conf>=0.60: bg,tc,obj=AMBERBG,AMBER,"badge_md"
        else:            bg,tc,obj=REDBG,RED,"badge_lo"
        bd=QLabel(f"{conf:.0%}"); bd.setObjectName(obj); bd.setFixedWidth(44)
        bd.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bd.setStyleSheet(f"background:{bg}; color:{tc}; border-radius:4px; "
                         f"padding:2px 0; font-size:11px; font-weight:700;")
        # info
        iv=QVBoxLayout(); iv.setSpacing(1); iv.setContentsMargins(0,0,0,0)
        fn=QLabel(os.path.basename(path)); fn.setObjectName("res_name")
        fn.setStyleSheet(f"font-size:12px; font-weight:600; color:{T1};")
        em=QLabel("Enhanced ✓" if enhanced else "Detection only")
        em.setObjectName("res_sub")
        em.setStyleSheet(f"font-size:10px; color:{GREEN if enhanced else T3};")
        iv.addWidget(fn); iv.addWidget(em)
        rl.addWidget(bd); rl.addLayout(iv,1)
        self._ril.insertWidget(self._ril.count()-1, rw)
        QTimer.singleShot(30, lambda: self._rsc.verticalScrollBar().setValue(
            self._rsc.verticalScrollBar().maximum()))

    def _on_done(self,ok,msg):
        self._btn_run.setEnabled(True)
        if ok:
            self._set_chip("COMPLETE",GREEN,f"background:{GREENBG}; border:1px solid #1a5c34;")
            self._log.add(f"Done — {msg}","ok")
        else:
            self._set_chip("ERROR",RED,f"background:{REDBG}; border:1px solid #5c1a1a;")
            self._log.add(f"Failed — {msg}","err")

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            from ultralytics import YOLO
            self._log.add("Loading YOLOv8...","info")
            self._yolo=YOLO(MODEL_PATH)
            self._log.add("YOLOv8 ready","ok")
            import tensorflow as tf
            self._log.add("Loading blind deblur...","info")
            self._ang=tf.keras.models.load_model(ANGLE_MDL)
            self._lng=tf.keras.models.load_model(LENGTH_MDL)
            self._log.add("Blind deblur ready","ok")
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            import torch
            self._log.add("Loading ESRGAN v3...","info")
            m=SRVGGNetCompact(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=32,upscale=4,act_type='prelu')
            self._sr=RealESRGANer(scale=4,model_path=ESRGAN_PATH,model=m,tile=0,tile_pad=10,
                                   pre_pad=0,half=True,device=torch.device('cuda'))
            self._log.add("ESRGAN v3 ready","ok")
            self._ready=True
            self._set_chip("READY",GREEN,f"background:{GREENBG}; border:1px solid #1a5c34;")
            self._log.add("All systems operational","ok")
            if self._file: self._btn_run.setEnabled(True)
        except Exception as e:
            self._log.add(f"Load error: {e}","err")
            self._set_chip("ERROR",RED,f"background:{REDBG}; border:1px solid #5c1a1a;")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _reset_prog(self):
        self._pbar.setValue(0); self._ppct.setText("0%"); self._pmsg.setText("Waiting for input...")

    def _clear_results(self):
        while self._ril.count():
            it=self._ril.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._rph=QLabel("No results yet")
        self._rph.setStyleSheet(f"color:{T3}; font-size:11px;")
        self._rph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ril.addWidget(self._rph); self._ril.addStretch()
        self._r_count.setText("0 results")

    def _set_chip(self, text, color, bg_style):
        self._chip.setText(f"● {text}")
        self._chip.setStyleSheet(
            f"{bg_style}; color:{color}; border:none;"
            f"border-radius:10px; padding:3px 12px; font-size:10px; font-weight:600;"
        )

    def _open_out(self):
        import subprocess
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background:{SURF2}; border:1px solid {B1}; border-radius:6px; padding:4px; }}"
            f"QMenu::item {{ color:{T1}; padding:6px 20px; font-size:12px; border-radius:4px; }}"
            f"QMenu::item:selected {{ background:{SURF3}; }}"
        )
        a1 = menu.addAction("Open Detected Folder")
        a2 = menu.addAction("Open Enhanced Folder")
        a3 = menu.addAction("Open Both Folders")
        chosen = menu.exec(self._btn_out.mapToGlobal(self._btn_out.rect().bottomLeft()))
        if chosen == a1:
            subprocess.Popen(f'explorer "{OUT_DETECT}"')
        elif chosen == a2:
            subprocess.Popen(f'explorer "{OUT_ENHANCE}"')
        elif chosen == a3:
            subprocess.Popen(f'explorer "{OUT_DETECT}"')
            subprocess.Popen(f'explorer "{OUT_ENHANCE}"')

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop(); self._worker.wait(2000)
        e.accept()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app=QApplication(sys.argv)
    app.setStyle("Fusion")
    p=QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(BG))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(T1))
    p.setColor(QPalette.ColorRole.Base,            QColor(SURF1))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(SURF2))
    p.setColor(QPalette.ColorRole.Text,            QColor(T1))
    p.setColor(QPalette.ColorRole.Button,          QColor(SURF1))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(T1))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(p)
    w=App(); w.show()
    sys.exit(app.exec())