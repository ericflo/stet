package main

// Where the server's files live: editor bundles, edit scripts and the season's downloads.
// A directory in development; a bucket (Backblaze B2 through its S3 API) in production.
// Everything stored is immutable and named by its hash, so there is nothing to coordinate.

import (
	"bytes"
	"container/list"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/aws/smithy-go"
)

var errNoBlob = errors.New("no such file")

type Blobs interface {
	Put(ctx context.Context, key string, data []byte, contentType string) error
	// Open returns the file and its size. The caller closes it.
	Open(ctx context.Context, key string) (io.ReadCloser, int64, error)
	Describe() string
}

func bundleKey(id string) string  { return "bundles/" + id + ".tar.zst" }
func scriptKey(sha string) string { return "scripts/" + sha[:2] + "/" + sha + ".zst" }

// --- a directory ------------------------------------------------------------------------------

type dirBlobs struct{ root string }

func (d dirBlobs) Describe() string { return "directory " + d.root }

func (d dirBlobs) path(key string) (string, error) {
	if key == "" || strings.Contains(key, "..") || strings.HasPrefix(key, "/") {
		return "", errNoBlob
	}
	return filepath.Join(d.root, filepath.FromSlash(key)), nil
}

func (d dirBlobs) Put(ctx context.Context, key string, data []byte, contentType string) error {
	p, err := d.path(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".put-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), p)
}

func (d dirBlobs) Open(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	p, err := d.path(key)
	if err != nil {
		return nil, 0, err
	}
	f, err := os.Open(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, 0, errNoBlob
	} else if err != nil {
		return nil, 0, err
	}
	st, err := f.Stat()
	if err != nil || st.IsDir() {
		f.Close()
		return nil, 0, errNoBlob
	}
	return f, st.Size(), nil
}

// --- a bucket ---------------------------------------------------------------------------------

type S3Config struct {
	Endpoint, Region, Bucket, Prefix, KeyID, Secret string
}

type s3Blobs struct {
	client *s3.Client
	bucket string
	prefix string
}

func newS3Blobs(cfg S3Config) *s3Blobs {
	client := s3.New(s3.Options{
		Region:       cfg.Region,
		BaseEndpoint: aws.String(cfg.Endpoint),
		UsePathStyle: true,
		Credentials:  credentials.NewStaticCredentialsProvider(cfg.KeyID, cfg.Secret, ""),
		// Backblaze B2 rejects the SDK's default flexible checksum headers on some paths.
		RequestChecksumCalculation: aws.RequestChecksumCalculationWhenRequired,
		ResponseChecksumValidation: aws.ResponseChecksumValidationWhenRequired,
	})
	return &s3Blobs{client: client, bucket: cfg.Bucket, prefix: cfg.Prefix}
}

func (b *s3Blobs) Describe() string { return "bucket " + b.bucket + "/" + b.prefix }

func (b *s3Blobs) Put(ctx context.Context, key string, data []byte, contentType string) error {
	_, err := b.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(b.bucket), Key: aws.String(b.prefix + key), Body: bytes.NewReader(data),
		ContentLength: aws.Int64(int64(len(data))), ContentType: aws.String(contentType),
	})
	return err
}

func (b *s3Blobs) Open(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	out, err := b.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(b.bucket), Key: aws.String(b.prefix + key)})
	if err != nil {
		var nsk *types.NoSuchKey
		var ae smithy.APIError
		if errors.As(err, &nsk) || (errors.As(err, &ae) && (ae.ErrorCode() == "NoSuchKey" || ae.ErrorCode() == "NotFound" || ae.ErrorCode() == "404")) {
			return nil, 0, errNoBlob
		}
		return nil, 0, err
	}
	return out.Body, aws.ToInt64(out.ContentLength), nil
}

// --- a small cache in front -------------------------------------------------------------------

// blobCache keeps recently used small files in memory: the site's source viewer opens the
// same bundle again and again, and a bucket round trip per page view would be silly.
type blobCache struct {
	Blobs
	mu    sync.Mutex
	max   int64
	size  int64
	order *list.List // front = most recent; values are *cached
	byKey map[string]*list.Element
}

type cached struct {
	key  string
	data []byte
}

const maxCachedBlob = 8 << 20

func newBlobCache(inner Blobs, max int64) *blobCache {
	return &blobCache{Blobs: inner, max: max, order: list.New(), byKey: map[string]*list.Element{}}
}

func (c *blobCache) remember(key string, data []byte) {
	if int64(len(data)) > maxCachedBlob {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.byKey[key]; ok {
		return
	}
	c.byKey[key] = c.order.PushFront(&cached{key, data})
	c.size += int64(len(data))
	for c.size > c.max && c.order.Len() > 1 {
		old := c.order.Remove(c.order.Back()).(*cached)
		delete(c.byKey, old.key)
		c.size -= int64(len(old.data))
	}
}

// Read returns a whole file, up to limit bytes.
func (c *blobCache) Read(ctx context.Context, key string, limit int64) ([]byte, error) {
	c.mu.Lock()
	if el, ok := c.byKey[key]; ok {
		c.order.MoveToFront(el)
		data := el.Value.(*cached).data
		c.mu.Unlock()
		return data, nil
	}
	c.mu.Unlock()
	rc, size, err := c.Blobs.Open(ctx, key)
	if err != nil {
		return nil, err
	}
	defer rc.Close()
	if size > limit {
		return nil, errors.New("the file is larger than expected")
	}
	data, err := io.ReadAll(io.LimitReader(rc, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errors.New("the file is larger than expected")
	}
	c.remember(key, data)
	return data, nil
}

func (c *blobCache) Put(ctx context.Context, key string, data []byte, contentType string) error {
	if err := c.Blobs.Put(ctx, key, data, contentType); err != nil {
		return err
	}
	c.remember(key, data)
	return nil
}
