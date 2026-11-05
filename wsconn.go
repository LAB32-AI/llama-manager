package main

import (
	"bufio"
	"context"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"
)

// A minimal WebSocket server (RFC 6455) for the Responses API's WebSocket
// mode: text messages, fragmentation, ping/pong and close. The Go standard
// library has none.

const (
	wsGUID       = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
	wsMaxMessage = 64 << 20
	wsPingEvery  = 30 * time.Second
	wsIdle       = 90 * time.Second // no frame (not even a pong) for this long: gone
	opCont       = 0x0
	opText       = 0x1
	opBinary     = 0x2
	opClose      = 0x8
	opPing       = 0x9
	opPong       = 0xA
)

type wsConn struct {
	c      net.Conn
	br     *bufio.Reader
	wmu    sync.Mutex
	closed atomic.Bool // a close frame went out
}

var errWSVersion = errors.New("WebSocket version 13 is required")

// pickProtocol chooses the subprotocol to answer: the first one offered
// that isn't an API key, else the first one. A browser that offered any
// drops a connection that picks none.
func pickProtocol(offered []string) string {
	for _, p := range offered {
		if !strings.HasPrefix(p, wsKeyProtocol) {
			return p
		}
	}
	if len(offered) > 0 {
		return offered[0]
	}
	return ""
}

// wsAccept completes a client's WebSocket handshake, naming protocol as the
// subprotocol when it isn't empty.
func wsAccept(w http.ResponseWriter, r *http.Request, protocol string) (*wsConn, error) {
	if !strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade") || !isWebSocket(r) {
		return nil, errors.New("not a WebSocket handshake")
	}
	if r.Header.Get("Sec-WebSocket-Version") != "13" {
		return nil, errWSVersion
	}
	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		return nil, errors.New("Sec-WebSocket-Key is missing")
	}
	sum := sha1.Sum([]byte(key + wsGUID))
	conn, brw, err := http.NewResponseController(w).Hijack()
	if err != nil {
		return nil, err
	}
	resp := "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " +
		base64.StdEncoding.EncodeToString(sum[:]) + "\r\n"
	if protocol != "" {
		resp += "Sec-WebSocket-Protocol: " + protocol + "\r\n"
	}
	if _, err := io.WriteString(conn, resp+"\r\n"); err != nil {
		conn.Close()
		return nil, err
	}
	_ = conn.SetDeadline(time.Time{})
	return &wsConn{c: conn, br: brw.Reader}, nil
}

func (c *wsConn) writeFrame(op byte, payload []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	hdr := []byte{0x80 | op}
	switch n := len(payload); {
	case n < 126:
		hdr = append(hdr, byte(n))
	case n <= 0xFFFF:
		hdr = append(hdr, 126, byte(n>>8), byte(n))
	default:
		hdr = append(hdr, 127)
		hdr = binary.BigEndian.AppendUint64(hdr, uint64(n))
	}
	_ = c.c.SetWriteDeadline(time.Now().Add(30 * time.Second))
	if _, err := c.c.Write(append(hdr, payload...)); err != nil {
		return err
	}
	return nil
}

func (c *wsConn) WriteText(b []byte) error { return c.writeFrame(opText, b) }

// keepAlive pings until ctx ends, so a peer that vanished is noticed: its
// reads time out after wsIdle without a pong.
func (c *wsConn) keepAlive(ctx context.Context) {
	t := time.NewTicker(wsPingEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if c.writeFrame(opPing, nil) != nil {
				return
			}
		}
	}
}

// fail closes the connection for a protocol error.
func (c *wsConn) fail(code uint16, err error) error {
	c.closeWith(code, nil)
	return err
}

// ReadMessage returns the next whole data message, answering pings on the
// way. A close frame (or a broken connection) ends it with an error.
func (c *wsConn) ReadMessage() ([]byte, error) {
	var msg []byte
	started, text := false, false
	for {
		_ = c.c.SetReadDeadline(time.Now().Add(wsIdle))
		var h [2]byte
		if _, err := io.ReadFull(c.br, h[:]); err != nil {
			return nil, err
		}
		fin, op := h[0]&0x80 != 0, h[0]&0x0F
		if h[0]&0x70 != 0 {
			return nil, c.fail(1002, errors.New("reserved bits set: no extension was agreed"))
		}
		if op >= opClose && (!fin || h[1]&0x7F > 125) {
			return nil, c.fail(1002, errors.New("control frames must be whole and at most 125 bytes"))
		}
		masked := h[1]&0x80 != 0
		n := uint64(h[1] & 0x7F)
		switch n {
		case 126:
			var b [2]byte
			if _, err := io.ReadFull(c.br, b[:]); err != nil {
				return nil, err
			}
			n = uint64(binary.BigEndian.Uint16(b[:]))
		case 127:
			var b [8]byte
			if _, err := io.ReadFull(c.br, b[:]); err != nil {
				return nil, err
			}
			n = binary.BigEndian.Uint64(b[:])
		}
		if !masked {
			return nil, c.fail(1002, errors.New("client frames must be masked"))
		}
		if n > wsMaxMessage || uint64(len(msg))+n > wsMaxMessage {
			return nil, c.fail(1009, fmt.Errorf("message over %d MB", wsMaxMessage>>20))
		}
		var mask [4]byte
		if _, err := io.ReadFull(c.br, mask[:]); err != nil {
			return nil, err
		}
		payload := make([]byte, n)
		if _, err := io.ReadFull(c.br, payload); err != nil {
			return nil, err
		}
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
		switch op {
		case opPing:
			_ = c.writeFrame(opPong, payload)
			continue
		case opPong:
			continue
		case opClose:
			// Echo the peer's status code, as RFC 6455 asks.
			if len(payload) >= 2 {
				c.closeWith(binary.BigEndian.Uint16(payload), nil)
			} else {
				c.closeWith(1000, nil)
			}
			return nil, io.EOF
		case opText, opBinary:
			if started {
				return nil, c.fail(1002, errors.New("a new message began inside a fragmented one"))
			}
			started, msg, text = true, payload, op == opText
		case opCont:
			if !started {
				return nil, c.fail(1002, errors.New("continuation without a message"))
			}
			msg = append(msg, payload...)
		default:
			return nil, c.fail(1002, fmt.Errorf("unknown opcode %d", op))
		}
		if fin {
			if text && !utf8.Valid(msg) {
				return nil, c.fail(1007, errors.New("text message isn't UTF-8"))
			}
			return msg, nil
		}
	}
}

// closeWith sends one close frame; later calls do nothing.
func (c *wsConn) closeWith(code uint16, _ error) {
	if c.closed.Swap(true) {
		return
	}
	_ = c.writeFrame(opClose, binary.BigEndian.AppendUint16(nil, code))
}

func (c *wsConn) Close() error {
	c.closeWith(1000, nil)
	return c.c.Close()
}
