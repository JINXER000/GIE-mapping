

#ifndef SRC_SIMPLE_LOGGER_H
#define SRC_SIMPLE_LOGGER_H

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include "iomanip"
using  namespace std;

class csvfile;

inline static csvfile& endrow(csvfile& file);
inline static csvfile& flush(csvfile& file);

class csvfile
{
    std::ofstream fs_;
    const std::string separator_;
public:
    csvfile(const std::string filename, const std::string separator = ",")
            : fs_()
            , separator_(separator)
    {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        fs_.open(filename);
    }

    ~csvfile()
    {
        flush();
        fs_.close();
    }

    void flush()
    {
        fs_.flush();
    }

    void endrow()
    {
        fs_ << std::endl;
    }

    csvfile& operator << ( csvfile& (* val)(csvfile&))
    {
        return val(*this);
    }

    csvfile& operator << (const char * val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    csvfile& operator << (const std::string & val)
    {
        fs_ << '"' << val << '"' << separator_;
        return *this;
    }

    template<typename T>
    csvfile& operator << (const T& val)
    {
        fs_ << val << separator_;
        return *this;
    }
};


inline static csvfile& endrow(csvfile& file)
{
    file.endrow();
    return file;
}

inline static csvfile& flush(csvfile& file)
{
    file.flush();
    return file;
}

#endif //SRC_SIMPLE_LOGGER_H
