import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import paramiko
from pathlib import Path


# Palette
WHITE       = "#F5F7FA"
MARINE      = "#1B3A6B"
MARINE_LT   = "#2C5282"
MARINE_DIM  = "#E8EDF5"
ACCENT      = "#E8A020"       
TEXT_DARK   = "#0D1B2A"
TEXT_MID    = "#4A5568"
TEXT_LIGHT  = "#A0AEC0"
BORDER      = "#CBD5E0"
SUCCESS     = "#2F855A"
ERROR_RED   = "#C53030"

FONT_HEAD   = ("Concert One", 22, "normal")
FONT_SUB    = ("Concert One", 13, "normal")
FONT_LABEL  = ("Concert One", 10, "normal")
FONT_BTN    = ("Concert One", 11, "normal")
FONT_SMALL  = ("Concert One", 9,  "normal")
FONT_MONO   = ("Courier New",  9,  "normal")


#Reusable widgets
def styled_entry(parent, show=None, **kw):
    e = tk.Entry(
        parent,
        font=("Helvetica", 11),
        bg=WHITE,
        fg=TEXT_DARK,
        insertbackground=MARINE,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=MARINE,
        show=show or "",
        **kw,
    )
    return e


def marine_btn(parent, text, command, width=18, accent=False):
    bg  = ACCENT   if accent else MARINE
    hov = "#C8870F" if accent else MARINE_LT

    # Use Label instead of Button — tk.Button bg/fg is overridden by the
    # system theme on Linux/macOS (dark mode), making it unreadable.
    # Labels always honour the colours we set.
    btn = tk.Label(
        parent,
        text=text,
        font=FONT_BTN,
        bg=bg,
        fg=WHITE,
        relief="flat",
        bd=0,
        padx=14,
        pady=8,
        cursor="hand2",
        width=width,
    )

    btn._disabled  = False
    btn._cmd       = command
    btn._bg_normal = bg
    btn._bg_hover  = hov

    def _click(e):
        if not btn._disabled:
            btn._cmd()

    def _enter(e):
        if not btn._disabled:
            btn.config(bg=btn._bg_hover)

    def _leave(e):
        if not btn._disabled:
            btn.config(bg=btn._bg_normal)

    btn.bind("<Button-1>", _click)
    btn.bind("<Enter>",    _enter)
    btn.bind("<Leave>",    _leave)

    # Patch .config() so callers can do btn.config(state="disabled"/"normal", text=..., command=...)
    _real_config = btn.config
    def _config(**kw):
        if "state" in kw:
            s = kw.pop("state")
            btn._disabled = (s == "disabled")
            _real_config(
                fg="#aaaaaa"    if btn._disabled else WHITE,
                bg=BORDER       if btn._disabled else btn._bg_normal,
            )
        if "command" in kw:
            btn._cmd = kw.pop("command")
        if kw:
            _real_config(**kw)
    btn.config = _config  # type: ignore

    return btn


def ghost_btn(parent, text, command, width=14):
    btn = tk.Button(
        parent,
        text=text,
        font=FONT_BTN,
        bg=MARINE_DIM,
        fg=MARINE,
        activebackground=BORDER,
        activeforeground=MARINE,
        relief="flat",
        bd=0,
        padx=12,
        pady=7,
        cursor="hand2",
        width=width,
        command=command,
    )
    return btn


def section_label(parent, text):
    return tk.Label(
        parent,
        text=text,
        font=FONT_LABEL,
        bg=WHITE,
        fg=TEXT_MID,
    )


def divider(parent):
    return tk.Frame(parent, height=1, bg=BORDER)


# Main App

class AILabTransfer(tk.Tk):
    SERVERS = [
        "ailab-fe01.srv.aau.dk",
        "ailab-fe02.srv.aau.dk",
    ]

    def __init__(self):
        super().__init__()
        self.title("AILab Transfer")
        self.resizable(False, False)
        self.configure(bg=WHITE)
        self._center(560, 520)

        # State
        self.server_var    = tk.StringVar(value=self.SERVERS[0])
        self.username_var  = tk.StringVar()
        self.password_var  = tk.StringVar()
        self.direction_var    = tk.StringVar(value="to_ailab")   # or "from_ailab"
        self.local_paths      = []          # list of local file paths
        self.remote_path      = tk.StringVar(value="/home/")
        self.use_project_path = tk.BooleanVar(value=False)
        self.PROJECT_PATH     = "/ceph/project/klnrs"
        self.ssh              = None

        self._build_header()
        self._show_server_page()

    # Layout helpers

    def _center(self, w, h):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _build_header(self):
        hdr = tk.Frame(self, bg=MARINE, height=64)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr, text="AILab", font=("Concert One", 20),
            bg=MARINE, fg=WHITE
        ).pack(side="left", padx=20, pady=12)

        tk.Label(
            hdr, text="File Transfer", font=("Concert One", 13),
            bg=MARINE, fg=ACCENT
        ).pack(side="left", pady=16)

        # Breadcrumb / step indicator on the right
        self.step_label = tk.Label(
            hdr, text="", font=FONT_SMALL,
            bg=MARINE, fg=TEXT_LIGHT
        )
        self.step_label.pack(side="right", padx=20)

    def _clear_body(self):
        for w in self.winfo_children():
            if w.cget("bg") != MARINE:   # leave header alone
                w.destroy()

    def _set_step(self, step, total=5):
        self.step_label.config(text=f"Step {step} of {total}")

    def _card(self, pady=(24, 16)):
        card = tk.Frame(self, bg=WHITE, padx=32, pady=18)
        card.pack(fill="both", expand=True, padx=24, pady=pady)
        return card

    # Page 1 : Server selection
    def _show_server_page(self):
        self._clear_body()
        self._set_step(1)
        card = self._card()

        tk.Label(card, text="Select Server", font=FONT_HEAD,
                 bg=WHITE, fg=MARINE).pack(anchor="w")
        tk.Label(card, text="Choose an AILab front-end node to connect to.",
                 font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 18))

        divider(card).pack(fill="x", pady=(0, 20))

        for srv in self.SERVERS:
            row = tk.Frame(card, bg=WHITE)
            row.pack(fill="x", pady=6)

            rb = tk.Radiobutton(
                row,
                text=srv,
                variable=self.server_var,
                value=srv,
                font=("Helvetica", 11),
                bg=WHITE,
                fg=TEXT_DARK,
                activebackground=WHITE,
                selectcolor=MARINE_DIM,
                relief="flat",
                cursor="hand2",
            )
            rb.pack(anchor="w", padx=6)

        spacer = tk.Frame(card, bg=WHITE, height=20)
        spacer.pack()

        btn_row = tk.Frame(card, bg=WHITE)
        btn_row.pack(fill="x")
        marine_btn(btn_row, "Connect  →", self._show_login_page, accent=True).pack(side="right")

    # Page 2 : Login
    def _show_login_page(self):
        self._clear_body()
        self._set_step(2)
        card = self._card()

        tk.Label(card, text="Login", font=FONT_HEAD,
                 bg=WHITE, fg=MARINE).pack(anchor="w")
        tk.Label(card, text=f"Authenticate with {self.server_var.get()}",
                 font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 18))

        divider(card).pack(fill="x", pady=(0, 20))

        # Username
        section_label(card, "AAU USERNAME").pack(anchor="w")
        self._user_entry = styled_entry(card, textvariable=self.username_var)
        self._user_entry.pack(fill="x", ipady=7, pady=(4, 14))
        self._user_entry.focus()

        # Password
        section_label(card, "PASSWORD").pack(anchor="w")
        self._pass_entry = styled_entry(card, show="●", textvariable=self.password_var)
        self._pass_entry.pack(fill="x", ipady=7, pady=(4, 4))
        self._pass_entry.bind("<Return>", lambda e: self._do_login())

        self._login_status = tk.Label(card, text="", font=FONT_SMALL,
                                      bg=WHITE, fg=ERROR_RED)
        self._login_status.pack(anchor="w", pady=(4, 12))

        btn_row = tk.Frame(card, bg=WHITE)
        btn_row.pack(fill="x")
        ghost_btn(btn_row, "←  Back", self._show_server_page).pack(side="left")
        marine_btn(btn_row, "Sign In  →", self._do_login, accent=True).pack(side="right")

    def _do_login(self):
        self._login_status.config(text="Connecting…", fg=TEXT_MID)
        self.update_idletasks()

        username = self.username_var.get().strip()
        password = self.password_var.get()

        if not username or not password:
            self._login_status.config(text="Please enter username and password.", fg=ERROR_RED)
            return

        def attempt():
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(
                    self.server_var.get(),
                    username=username,
                    password=password,
                    timeout=10,
                    look_for_keys=False,
                    allow_agent=False,
                )
                self.ssh = client
                self.after(0, self._show_direction_page)
            except paramiko.AuthenticationException:
                self.after(0, lambda: self._login_status.config(
                    text="Authentication failed. Check credentials.", fg=ERROR_RED))
            except Exception as exc:
                self.after(0, lambda: self._login_status.config(
                    text=f"Connection error: {exc}", fg=ERROR_RED))

        threading.Thread(target=attempt, daemon=True).start()

    # Page 3 : Direction
    def _show_direction_page(self):
        self._clear_body()
        self._set_step(3)
        card = self._card()

        tk.Label(card, text="Transfer Direction", font=FONT_HEAD,
                 bg=WHITE, fg=MARINE).pack(anchor="w")
        tk.Label(card, text=f"Logged in as {self.username_var.get()} @ {self.server_var.get()}",
                 font=FONT_SMALL, bg=WHITE, fg=SUCCESS).pack(anchor="w", pady=(2, 18))

        divider(card).pack(fill="x", pady=(0, 24))

        for val, icon, title, desc in [
            ("to_ailab",   "⬆",  "Personal  →  AILab",   "Upload files from your PC to the server"),
            ("from_ailab", "⬇",  "AILab  →  Personal",   "Download files from the server to your PC"),
        ]:
            frame = tk.Frame(card, bg=MARINE_DIM, bd=0, relief="flat",
                             cursor="hand2", padx=16, pady=14)
            frame.pack(fill="x", pady=8)

            inner = tk.Frame(frame, bg=MARINE_DIM)
            inner.pack(fill="x")

            rb = tk.Radiobutton(
                inner, variable=self.direction_var, value=val,
                bg=MARINE_DIM, activebackground=MARINE_DIM,
                selectcolor=WHITE, relief="flat", cursor="hand2",
            )
            rb.pack(side="left")

            tk.Label(inner, text=icon, font=("Helvetica", 18),
                     bg=MARINE_DIM, fg=MARINE).pack(side="left", padx=(4, 10))

            txt = tk.Frame(inner, bg=MARINE_DIM)
            txt.pack(side="left")
            tk.Label(txt, text=title, font=FONT_SUB,
                     bg=MARINE_DIM, fg=MARINE).pack(anchor="w")
            tk.Label(txt, text=desc, font=FONT_SMALL,
                     bg=MARINE_DIM, fg=TEXT_MID).pack(anchor="w")

            frame.bind("<Button-1>", lambda e, v=val: self.direction_var.set(v))
            inner.bind("<Button-1>", lambda e, v=val: self.direction_var.set(v))

        btn_row = tk.Frame(card, bg=WHITE)
        btn_row.pack(fill="x", pady=(20, 0))
        ghost_btn(btn_row, "←  Back", self._show_login_page).pack(side="left")
        marine_btn(btn_row, "Next  →", self._show_file_page, accent=True).pack(side="right")

    # Page 4 : File selection
    def _show_file_page(self):
        self._clear_body()
        self._set_step(4)
        card = self._card(pady=(16, 8))
        self.local_paths = []

        direction = self.direction_var.get()

        if direction == "to_ailab":
            tk.Label(card, text="Select Files to Upload", font=FONT_HEAD,
                     bg=WHITE, fg=MARINE).pack(anchor="w")
            tk.Label(card, text="Choose local files to send to AILab.",
                     font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 10))
        else:
            tk.Label(card, text="Select Remote Files", font=FONT_HEAD,
                     bg=WHITE, fg=MARINE).pack(anchor="w")
            tk.Label(card, text="Enter remote file paths (one per line) to download.",
                     font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 10))

        divider(card).pack(fill="x", pady=(0, 10))

        # File list box
        list_frame = tk.Frame(card, bg=WHITE, bd=1, relief="flat",
                              highlightthickness=1, highlightbackground=BORDER)
        list_frame.pack(fill="both", expand=True)

        self._file_listbox = tk.Listbox(
            list_frame, font=FONT_MONO, bg=WHITE, fg=TEXT_DARK,
            selectbackground=MARINE_DIM, selectforeground=MARINE,
            relief="flat", bd=0, height=7,
        )
        self._file_listbox.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self._file_listbox.yview)
        sb.pack(side="right", fill="y")
        self._file_listbox.config(yscrollcommand=sb.set)

        # Buttons below list
        ctrl = tk.Frame(card, bg=WHITE)
        ctrl.pack(fill="x", pady=(8, 0))

        if direction == "to_ailab":
            ghost_btn(ctrl, "+ Add Files", self._browse_local_files, width=12).pack(side="left", padx=(0, 6))
            ghost_btn(ctrl, "+ Add Folder", self._browse_local_folder, width=12).pack(side="left")
        else:
            ghost_btn(ctrl, "+ Browse Remote", self._browse_remote, width=14).pack(side="left")

        ghost_btn(ctrl, "✕ Remove", self._remove_selected, width=10).pack(side="left", padx=6)

        btn_row = tk.Frame(card, bg=WHITE)
        btn_row.pack(fill="x", pady=(12, 0))
        ghost_btn(btn_row, "←  Back", self._show_direction_page).pack(side="left")
        marine_btn(btn_row, "Next  →", self._show_dest_page, accent=True).pack(side="right")

    def _browse_local_files(self):
        files = filedialog.askopenfilenames(title="Select files")
        for f in files:
            if f not in self.local_paths:
                self.local_paths.append(f)
                self._file_listbox.insert("end", f)

    def _browse_local_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if folder and folder not in self.local_paths:
            self.local_paths.append(folder)
            self._file_listbox.insert("end", folder)

    def _browse_remote(self):
        path = self._ask_string("Remote path", "Enter a remote file/folder path:")
        if path:
            self.local_paths.append(path)
            self._file_listbox.insert("end", path)

    def _remove_selected(self):
        sel = list(self._file_listbox.curselection())
        for i in reversed(sel):
            self._file_listbox.delete(i)
            del self.local_paths[i]

    def _ask_string(self, title, prompt):
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg=WHITE)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.geometry("380x130")

        tk.Label(dlg, text=prompt, font=FONT_LABEL, bg=WHITE, fg=TEXT_MID
                 ).pack(anchor="w", padx=16, pady=(14, 4))
        var = tk.StringVar()
        e = styled_entry(dlg, textvariable=var)
        e.pack(fill="x", padx=16, ipady=6)
        e.focus()

        result = [None]

        def ok():
            result[0] = var.get().strip()
            dlg.destroy()

        e.bind("<Return>", lambda _: ok())
        marine_btn(dlg, "OK", ok, width=10).pack(pady=10)
        dlg.wait_window()
        return result[0]

    # Page 5 : Destination & transfer

    def _show_dest_page(self):
        if not self.local_paths:
            messagebox.showwarning("No files", "Please add at least one file.")
            return

        self._clear_body()
        self._set_step(5)
        card = self._card(pady=(16, 8))

        direction = self.direction_var.get()

        if direction == "to_ailab":
            tk.Label(card, text="Remote Destination", font=FONT_HEAD,
                     bg=WHITE, fg=MARINE).pack(anchor="w")
            tk.Label(card, text="Where on AILab should the files be stored?",
                     font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 12))
        else:
            tk.Label(card, text="Local Destination", font=FONT_HEAD,
                     bg=WHITE, fg=MARINE).pack(anchor="w")
            tk.Label(card, text="Where on your PC should the files be saved?",
                     font=FONT_SMALL, bg=WHITE, fg=TEXT_MID).pack(anchor="w", pady=(2, 12))

        divider(card).pack(fill="x", pady=(0, 14))

        section_label(card, "DESTINATION PATH").pack(anchor="w")

        if direction == "to_ailab":
            proj_row = tk.Frame(card, bg=MARINE_DIM, bd=0, relief="flat", padx=12, pady=8)
            proj_row.pack(fill="x", pady=(4, 8))

            def _on_project_toggle():
                if self.use_project_path.get():
                    self.remote_path.set(self.PROJECT_PATH)
                    dest_entry.config(state="disabled", fg=TEXT_MID)
                else:
                    dest_entry.config(state="normal", fg=TEXT_DARK)

            tk.Checkbutton(
                proj_row,
                text=f"  Use project path:  {self.PROJECT_PATH}",
                variable=self.use_project_path,
                font=FONT_LABEL,
                bg=MARINE_DIM,
                fg=MARINE,
                activebackground=MARINE_DIM,
                selectcolor=WHITE,
                relief="flat",
                cursor="hand2",
                command=_on_project_toggle,
            ).pack(anchor="w")

        path_row = tk.Frame(card, bg=WHITE)
        path_row.pack(fill="x", pady=(4, 14))

        dest_entry = styled_entry(path_row, textvariable=self.remote_path)
        dest_entry.pack(side="left", fill="x", expand=True, ipady=7)

        if direction == "to_ailab" and self.use_project_path.get():
            dest_entry.config(state="disabled", fg=TEXT_MID)

        if direction == "from_ailab":
            def pick_local():
                d = filedialog.askdirectory()
                if d:
                    self.remote_path.set(d)
            ghost_btn(path_row, "Browse", pick_local, width=8).pack(side="left", padx=(8, 0))

        # Summary box
        section_label(card, "FILES TO TRANSFER").pack(anchor="w")
        summ = tk.Text(card, font=FONT_MONO, bg=MARINE_DIM, fg=TEXT_DARK,
                       relief="flat", bd=0, height=5, state="normal")
        summ.pack(fill="x", pady=(4, 0))
        for p in self.local_paths:
            summ.insert("end", p + "\n")
        summ.config(state="disabled")

        # Progress
        self._prog_var  = tk.DoubleVar()
        self._prog_lbl  = tk.Label(card, text="", font=FONT_SMALL,
                                   bg=WHITE, fg=TEXT_MID)
        self._prog_lbl.pack(anchor="w", pady=(10, 2))

        self._prog_bar = ttk.Progressbar(card, variable=self._prog_var,
                                         maximum=100, length=400)
        self._prog_bar.pack(fill="x")

        btn_row = tk.Frame(card, bg=WHITE)
        btn_row.pack(fill="x", pady=(14, 0))
        ghost_btn(btn_row, "←  Back", self._show_file_page).pack(side="left")
        self._transfer_btn = marine_btn(
            btn_row, "Start Transfer  ⬆" if direction == "to_ailab" else "Download  ⬇",
            self._start_transfer, accent=True, width=20,
        )
        self._transfer_btn.pack(side="right")

    def _start_transfer(self):
        dest = self.remote_path.get().strip()
        if not dest:
            messagebox.showwarning("No destination", "Please enter a destination path.")
            return

        self._transfer_btn.config(state="disabled")
        self._prog_lbl.config(text="Starting transfer…", fg=TEXT_MID)

        direction = self.direction_var.get()

        def run():
            try:
                sftp = self.ssh.open_sftp()
                paths = self.local_paths
                total = len(paths)

                for idx, path in enumerate(paths, 1):
                    name = os.path.basename(path.rstrip("/\\"))
                    self.after(0, lambda n=name: self._prog_lbl.config(
                        text=f"Transferring: {n}  ({idx}/{total})", fg=TEXT_MID))

                    if direction == "to_ailab":
                        remote_file = dest.rstrip("/") + "/" + name
                        if os.path.isdir(path):
                            self._upload_dir(sftp, path, remote_file)
                        else:
                            sftp.put(path, remote_file,
                                     callback=lambda s, t: self._prog_var.set(s / t * 100))
                    else:
                        local_file = os.path.join(dest, name)
                        sftp.get(path, local_file,
                                 callback=lambda s, t: self._prog_var.set(s / t * 100))

                    self._prog_var.set(idx / total * 100)

                sftp.close()
                self.after(0, self._transfer_done)
            except Exception as exc:
                self.after(0, lambda: self._transfer_error(str(exc)))

        threading.Thread(target=run, daemon=True).start()

    def _upload_dir(self, sftp, local_dir, remote_dir):
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass
        for item in os.listdir(local_dir):
            local_path  = os.path.join(local_dir, item)
            remote_path = remote_dir + "/" + item
            if os.path.isdir(local_path):
                self._upload_dir(sftp, local_path, remote_path)
            else:
                sftp.put(local_path, remote_path)

    def _transfer_done(self):
        self._prog_lbl.config(text="✔  Transfer complete!", fg=SUCCESS)
        self._transfer_btn.config(state="normal",
                                  text="Transfer More", command=self._show_direction_page)

    def _transfer_error(self, msg):
        self._prog_lbl.config(text=f"Error: {msg}", fg=ERROR_RED)
        self._transfer_btn.config(state="normal")


if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = AILabTransfer()
    app.mainloop()